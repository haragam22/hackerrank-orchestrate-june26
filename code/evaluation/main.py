"""
code/evaluation/main.py
Phase 9: Evaluation Script — Entry Point.

Compares the pipeline's output against the ground-truth labels in sample_claims.csv.
Computes accuracy for exact match fields and set-based F1 for risk_flags.
Writes the final evaluation_report.md to the repo root.

Run directly:
    python code/evaluation/main.py --pred output_sample.csv --truth dataset/sample_claims.csv
"""

import sys
import argparse
import pathlib
import pandas as pd

def compute_f1_score(pred_flags: set[str], true_flags: set[str]) -> float:
    # Ignore 'none' sentinel if present in either set
    pred_clean = {f for f in pred_flags if f and f != 'none'}
    true_clean = {f for f in true_flags if f and f != 'none'}
    
    if not pred_clean and not true_clean:
        return 1.0  # correctly predicted no flags
        
    true_positives = len(pred_clean & true_clean)
    if true_positives == 0:
        return 0.0
        
    precision = true_positives / len(pred_clean)
    recall = true_positives / len(true_clean)
    
    return 2 * (precision * recall) / (precision + recall)

def run_evaluation(pred_path: str, truth_path: str, report_out_path: str):
    print("=" * 60)
    print("Multi-Modal Evidence Review — Evaluation")
    print("=" * 60)

    try:
        df_pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
        df_true = pd.read_csv(truth_path, dtype=str, keep_default_na=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Merge on user_id
    df_merged = pd.merge(df_pred, df_true, on='user_id', suffixes=('_pred', '_true'))
    total = len(df_merged)
    
    if total == 0:
        print("ERROR: No matching user_ids found between predictions and truth.")
        return
        
    print(f"Evaluated {total} claims.\n")

    metrics = {
        'claim_status': 0,
        'severity': 0,
        'issue_type': 0,
        'object_part': 0,
        'evidence_standard_met': 0,
        'valid_image': 0,
    }
    
    f1_sum = 0.0
    failed_claims = []

    for _, row in df_merged.iterrows():
        # Exact match fields
        for field in metrics.keys():
            p = row[f'{field}_pred'].strip().lower()
            t = row[f'{field}_true'].strip().lower()
            if p == t:
                metrics[field] += 1
            elif field == 'claim_status':
                failed_claims.append({
                    'user_id': row['user_id'],
                    'predicted': p,
                    'expected': t,
                    'pred_just': row.get('claim_status_justification_pred', 'N/A'),
                    'true_just': row.get('claim_status_justification_true', 'N/A'),
                    'user_claim': row.get('user_claim_true', 'N/A'),
                    'pred_issue': row.get('issue_type_pred', 'N/A'),
                    'true_issue': row.get('issue_type_true', 'N/A'),
                    'pred_flags': row.get('risk_flags_pred', 'N/A'),
                    'true_flags': row.get('risk_flags_true', 'N/A'),
                })
                
        # Risk flags F1
        p_flags = set(f.strip().lower() for f in row['risk_flags_pred'].split(';'))
        t_flags = set(f.strip().lower() for f in row['risk_flags_true'].split(';'))
        f1_sum += compute_f1_score(p_flags, t_flags)

    # Compile results
    results = {}
    for field, correct in metrics.items():
        results[field] = correct / total
        
    results['risk_flags_f1'] = f1_sum / total

    # Print table
    print(f"{'Metric':<25} | {'Score':<10}")
    print("-" * 38)
    for field, score in results.items():
        print(f"{field:<25} | {score:.1%}")
    print("-" * 38)

    # Gate check
    claim_status_acc = results['claim_status']
    gate_passed = claim_status_acc >= 0.70
    gate_text = "✅ GATE PASSED (≥70%)" if gate_passed else "❌ GATE FAILED (<70%) — Revise prompts"
    
    print(f"\nTarget Metric (Claim Status): {claim_status_acc:.1%}")
    print(gate_text)
    
    if failed_claims:
        print("\nErrors in claim_status:")
        for err in failed_claims:
            print(f"- {err['user_id']}: Predicted '{err['predicted']}', Expected '{err['expected']}'")

    # Write Markdown Report
    report_md = f"""# Evaluation Report

## Summary
- **Total claims evaluated:** {total}
- **Gate Check:** {gate_text}

## Metrics

| Metric | Score |
|---|---|
| Claim Status Accuracy | **{results['claim_status']:.1%}** |
| Severity Accuracy | {results['severity']:.1%} |
| Issue Type Accuracy | {results['issue_type']:.1%} |
| Object Part Accuracy | {results['object_part']:.1%} |
| Evidence Standard Met Acc. | {results['evidence_standard_met']:.1%} |
| Valid Image Accuracy | {results['valid_image']:.1%} |
| Risk Flags (F1 Score) | {results['risk_flags_f1']:.1%} |

"""
    if failed_claims:
        report_md += "## Error Analysis (Claim Status)\n\n"
        report_md += "Detailed breakdown of claims where the predicted status did not match the ground truth:\n\n"
        for err in failed_claims:
            report_md += f"### Claim: `{err['user_id']}`\n"
            report_md += f"- **Status Match:** ❌ Predicted `{err['predicted']}`, Expected `{err['expected']}`\n"
            report_md += f"- **Issue Type:** Predicted `{err['pred_issue']}`, Expected `{err['true_issue']}`\n"
            report_md += f"- **Risk Flags:** Predicted `{err['pred_flags']}`, Expected `{err['true_flags']}`\n"
            
            # Use blockquotes for longer text
            report_md += f"\n**User Claim Transcript:**\n> {err['user_claim']}\n\n"
            report_md += f"**Predicted Justification (Agent 4):**\n> {err['pred_just']}\n\n"
            report_md += f"**Expected Justification (Ground Truth):**\n> {err['true_just']}\n\n"
            
            # Actionable insight generator
            insight = "🔍 **Improvement Insight:** "
            if err['predicted'] == 'not_enough_information' and err['expected'] in ['supported', 'contradicted']:
                insight += "The pipeline failed to confidently extract the required evidence. The Vision Model may have hallucinated poor image quality or missed details that the ground truth considered sufficient. Review Agent 2's validation criteria or Agent 3's strictness."
            elif err['predicted'] == 'contradicted' and err['expected'] == 'supported':
                insight += "The Vision Model completely missed the damage that was expected to be seen. This points to a hallucination in Agent 3 (Damage Assessor). Consider improving the few-shot examples or prompt instructions for the claimed object part."
            elif err['predicted'] == 'supported' and err['expected'] == 'contradicted':
                insight += "The Vision Model hallucinated damage that does not exist. Agent 3 is being too lenient or mistaking normal wear-and-tear / reflections for damage. Tune Agent 3 to be more skeptical."
            else:
                insight += "Review the prompt alignment. There may be a discrepancy between what Agent 3 considers valid damage vs the ground truth standard."
                
            report_md += f"{insight}\n\n---\n\n"

    with open(report_out_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"\nReport written to {report_out_path}")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="output_sample.csv")
    parser.add_argument("--truth", default="dataset/sample_claims.csv")
    parser.add_argument("--report", default="evaluation_report.md")
    
    args = parser.parse_args()
    
    # Resolve relative to repo root
    base = pathlib.Path(__file__).parent.parent.parent  # code/evaluation/ -> code/ -> repo root
    
    run_evaluation(
        str(base / args.pred),
        str(base / args.truth),
        str(base / args.report)
    )
