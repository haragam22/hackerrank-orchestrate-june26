"""
code/main.py
Phase 8: Pipeline Integration — Entry Point.
Wires together all agents and stages, executing the full multi-modal pipeline
on the provided claims CSV.

Run directly:
    python code/main.py --input dataset/sample_claims.csv --output output_sample.csv
    python code/main.py --input dataset/claims.csv --output output.csv
"""

import os
import sys
import argparse
import logging
import pathlib
import time

from google import genai
from dotenv import load_dotenv

# Add core/ to path so we can import modules
sys.path.insert(0, str(pathlib.Path(__file__).parent / "core"))

from core import stage0_loader
from core import stage1_rag
from core import agent1_parser
from core import agent2_validator
from core import agent3_assessor
from core import agent4_assembler
from core.schemas import ClaimOutput, DamageAssessment

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(input_path: str, output_path: str, dataset_base_dir: str):
    """Run the complete 4-agent pipeline on the input CSV."""
    logger.info("=" * 60)
    logger.info(f"Starting Multi-Modal Evidence Review Pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # ── Stage 0: Load Data ───────────────────────────────────────
    logger.info("[Stage 0] Loading datasets...")
    base = pathlib.Path(dataset_base_dir)
    claims = stage0_loader.load_claims(input_path)

    user_history = stage0_loader.load_user_history(str(base / "user_history.csv"))
    evidence_reqs = stage0_loader.load_evidence_requirements(str(base / "evidence_requirements.csv"))
    
    # Load sample claims for RAG
    sample_claims = stage0_loader.load_sample_claims(str(base / "sample_claims.csv"))
    
    # Load clients
    load_dotenv(base.parent / ".env")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not gemini_key:
        logger.error("GEMINI_API_KEY not set in .env (needed for FAISS embeddings)")
        sys.exit(1)
    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY not set in .env (needed for OpenRouter models)")
        sys.exit(1)

    gemini_client = genai.Client(api_key=gemini_key)

    import openai
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    # ── Stage 1: Build RAG Index (with disk cache) ───────────────
    logger.info("[Stage 1] Building FAISS index for few-shot RAG...")
    cache_path = pathlib.Path(__file__).parent.parent / "rag_cache.npz"
    index_matrix = stage1_rag.build_faiss_index_cached(sample_claims, gemini_client, cache_path)

    # ── Process Claims ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Processing {len(claims)} claims...")
    logger.info("=" * 60)

    INTER_CLAIM_SLEEP = 15  # seconds between claims to stay within 15 RPM free-tier limit

    output_rows = []

    for i, claim_row in enumerate(claims, 1):
        uid = claim_row.get("user_id", "unknown")
        claim_object = claim_row.get("claim_object", "unknown")
        user_claim_text = claim_row.get("user_claim", "")

        logger.info(f"[{i}/{len(claims)}] Processing {uid} (Object: {claim_object})...")

        try:
            # 1. Retrieve history & evidence req
            history = user_history.get(uid, {})
            ev_req = stage0_loader.get_evidence_requirement(claim_object, evidence_reqs)

            # 2. Load Images
            images = stage0_loader.load_images(claim_row.get("image_paths", ""), dataset_base_dir)

            # 3. Agent 1: Parse Claim
            parsed = agent1_parser.run(user_claim_text, claim_object, openrouter_client)

            # 4. Agent 2: Validate Images
            validated = agent2_validator.run(images, claim_object, openrouter_client)

            # 5. Agent 3: Assess Damage
            few_shot = stage1_rag.retrieve_example(user_claim_text, index_matrix, sample_claims, gemini_client)
            assessed = agent3_assessor.run(images, claim_object, parsed, validated, ev_req, few_shot, openrouter_client)

            # 6. Agent 4: Risk & Assembly
            output = agent4_assembler.run(claim_row, parsed, validated, assessed, history, openrouter_client)

            output_rows.append(output.to_csv_row())

        except Exception as e:
            import traceback
            logger.error(f"  [!] Pipeline failed for {uid}: {e}")
            logger.error(traceback.format_exc())
            fallback_out = ClaimOutput.from_fallback(claim_row)
            output_rows.append(fallback_out.to_csv_row())

        # ── Checkpoint: write partial results every 5 claims ─────────
        if i % 5 == 0:
            stage0_loader.write_output_csv(output_rows, output_path + ".partial")
            logger.info(f"  Checkpoint saved ({i} claims processed).")

        # ── Rate-limit guard: sleep between claims ────────────────────
        if i < len(claims):
            logger.info(f"  Sleeping {INTER_CLAIM_SLEEP}s before next claim (rate limit guard)...")
            time.sleep(INTER_CLAIM_SLEEP)

    # ── Write Output ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Writing results...")
    stage0_loader.write_output_csv(output_rows, output_path)
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review Pipeline")
    parser.add_argument("--input", required=True, help="Path to input claims.csv")
    parser.add_argument("--output", required=True, help="Path to write output CSV")
    
    args = parser.parse_args()
    
    # We assume dataset/ is a sibling of the code/ directory where this runs
    dataset_dir = str(pathlib.Path(__file__).parent.parent / "dataset")
    
    run_pipeline(args.input, args.output, dataset_dir)
