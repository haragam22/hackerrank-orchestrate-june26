# IMPLEMENTATION.md
## Multi-Modal Evidence Review — Phased Build Plan

---

## Ground Rules

- **Never start a phase until the previous phase checkpoint passes.**
- **All prompt strings live in `prompts.py` only.** No prompt text in agent files.
- **No hardcoded expected outputs.** No `if claim_id == 'case_001'` anywhere.
- **Test each agent in isolation before wiring into `pipeline.py`.**
- **Each agent file is independently runnable** with a `if __name__ == "__main__"` test block.
- If something breaks, fix it in the current phase before moving forward.

---

## Phase 0 — Environment Setup
**Goal:** Repo cloned, dependencies installed, API key working, both text and vision calls confirmed.

### Tasks
- [ ] Clone the HackerRank repo
- [ ] Create `code/` and `evaluation/` directories
- [ ] Create `requirements.txt`:
  ```
  google-generativeai>=0.8.0
  pydantic>=2.0.0
  pandas>=2.0.0
  faiss-cpu>=1.7.4
  numpy>=1.24.0
  pillow>=10.0.0
  python-dotenv>=1.0.0
  ```
- [ ] `pip install -r requirements.txt`
- [ ] Create `.env`: `GEMINI_API_KEY=your_key_here`
- [ ] `smoke_test.py`: send text prompt to `gemini-2.5-flash`, print response
- [ ] `smoke_test.py`: send one image to `gemini-2.5-flash`, print response
- [ ] `smoke_test.py`: call `gemini-embedding-001` on a test string, print embedding length

### Checkpoint ✅
```bash
python code/smoke_test.py
# Expected: 3 successful responses printed, no errors
```

---

## Phase 1 — Schemas
**Goal:** `code/schemas.py` complete and self-tested. All agents import from here.

### Tasks
- [ ] Enums: `ClaimStatus`, `Severity`, `IssueType`, `CarPart`, `LaptopPart`, `PackagePart`, `RiskFlag`
- [ ] Internal agent output models:
  - `ParsedClaim` — Agent 1 output `{ object_part, issue_type, claim_summary }`
  - `ImageValidation` — Agent 2 output `{ valid_image, image_quality_flags[], per_image_notes }`
  - `DamageAssessment` — Agent 3 output `{ claim_status, issue_type, object_part, severity, supporting_image_ids[], justification, damage_flags[], evidence_standard_met, evidence_standard_met_reason }`
- [ ] Final output model: `ClaimOutput` — all 14 CSV columns
- [ ] Validators: bool → lowercase string, flags → semicolon-joined, image_ids → semicolon-joined
- [ ] `to_csv_row()` on `ClaimOutput` returning exact column-ordered dict
- [ ] Fallback constructors: `ParsedClaim.fallback()`, `ImageValidation.fallback()`, `DamageAssessment.fallback()`, `ClaimOutput.from_fallback(claim_row)`
- [ ] Self-test block: instantiate each model, assert serialization correct

### Checkpoint ✅
```bash
python code/schemas.py
# Expected: all self-test assertions pass, "schemas.py OK" printed
```

---

## Phase 2 — Infrastructure (Stage 0 + Stage 1)
**Goal:** All data loaded, FAISS index built, retrieval working.

### Tasks

**`code/stage0_loader.py`**
- [ ] `load_claims(path)` → `list[dict]`
- [ ] `load_user_history(path)` → `dict[str, dict]` keyed by user_id
- [ ] `load_evidence_requirements(path)` → `dict` keyed by `(claim_object, applies_to)`
- [ ] `load_sample_claims(path)` → `list[dict]` with all 14 output columns
- [ ] `load_images(image_paths_str, base_dir)` → `list[PIL.Image]` — splits semicolons, opens each; missing file → log + skip

**`code/stage1_rag.py`**
- [ ] `build_faiss_index(sample_claims, gemini_client)` → `(np.ndarray, list[dict])` — embed all `user_claim` texts, store matrix
- [ ] `retrieve_example(query_claim, index_matrix, sample_claims, gemini_client, threshold=0.75)` → `dict | None`
- [ ] Log similarity score for every retrieval

### Checkpoint ✅
```bash
python code/stage0_loader.py
# Expected: row counts printed for all 4 CSVs, no errors

python code/stage1_rag.py
# Expected: FAISS index built from 20 samples
#           3 test queries run, similarity scores printed
#           correct examples retrieved or None returned
```

---

## Phase 3 — Prompts
**Goal:** `code/prompts.py` complete with all 4 agent prompts. No prompt text anywhere else.

### Tasks

**`code/prompts.py`**
- [ ] `PROMPT_A_SYSTEM` + `build_prompt_a(user_claim, claim_object)` → Agent 1 (Claim Parser)
  - XML-wraps `user_claim`
  - Injects correct part enum for the claim_object
  - Demands JSON-only response
- [ ] `PROMPT_B_SYSTEM` + `build_prompt_b(claim_object)` → Agent 2 (Image Validator)
  - No claim details — just object type
  - Lists the 4 validation checks explicitly
  - Demands JSON-only response
- [ ] `PROMPT_C_SYSTEM` + `build_prompt_c(parsed_claim, evidence_req, few_shot_example)` → Agent 3 (Damage Assessor)
  - Injects `ParsedClaim` fields
  - Injects evidence requirement text
  - Injects few-shot example if not None
  - Enforces two-step LOCATE → ASSESS structure
  - Demands JSON-only response
- [ ] `PROMPT_D_SYSTEM` + `build_prompt_d(parsed_claim, image_validation, damage_assessment, user_history)` → Agent 4 (Risk & Assembly)
  - Injects all three prior agent outputs
  - Injects user history fields
  - States `claim_status` is fixed — cannot be changed
  - Demands JSON-only response
- [ ] `RETRY_SUFFIX` — appended on retry: "Your previous response was not valid JSON. Respond ONLY with the JSON object. No explanation, no markdown, no preamble."

### Checkpoint ✅
```bash
python code/prompts.py
# Expected: all 4 prompts printed with test data injected
#           visually verify XML wrapping, schema injection, few-shot injection
#           no missing fields, no broken f-strings
```

---

## Phase 4 — Agent 1: Claim Parser
**Goal:** Agent 1 running, parsing object_part and issue_type from conversation text.

### Tasks

**`code/agent1_parser.py`**
- [ ] `run(user_claim, claim_object, gemini_client)` → `ParsedClaim`
- [ ] Calls `build_prompt_a()` from prompts.py
- [ ] Sends text-only request to `gemini-2.5-flash`
- [ ] Parses JSON response into `ParsedClaim`
- [ ] On parse failure: retry with `RETRY_SUFFIX` appended
- [ ] On retry failure: return `ParsedClaim.fallback()`
- [ ] Logs raw response before parsing
- [ ] `time.sleep(1)` after every API call

### Checkpoint ✅
```bash
python code/agent1_parser.py
# Run against 5 sample claims (mix of car/laptop/package)
# Expected: correct object_part and issue_type for each
#           printed ParsedClaim objects, no crashes
```

---

## Phase 5 — Agent 2: Image Validator
**Goal:** Agent 2 running, correctly flagging image quality issues and validity.

### Tasks

**`code/agent2_validator.py`**
- [ ] `run(images, claim_object, gemini_client)` → `ImageValidation`
- [ ] Calls `build_prompt_b()` from prompts.py
- [ ] Sends images + text to `gemini-2.5-flash`
- [ ] Parses JSON into `ImageValidation`
- [ ] Validates all flags against `RiskFlag` enum — drops unknown values
- [ ] On parse failure: retry once
- [ ] On retry failure: return `ImageValidation.fallback()` with `valid_image=False`
- [ ] `time.sleep(1)` after every API call

### Checkpoint ✅
```bash
python code/agent2_validator.py
# Run against:
#   - 2 clean claims (valid images, right object)
#   - 1 claim with known blurry/quality issue (user_003 from sample)
#   - 1 claim with known mismatch (user_008 from sample, valid_image=false)
# Expected: correct valid_image bool for each
#           correct flags returned for quality issues
```

---

## Phase 6 — Agent 3: Damage Assessor
**Goal:** Agent 3 running, producing correct claim_status grounded in images.

### Tasks

**`code/agent3_assessor.py`**
- [ ] `run(images, parsed_claim, evidence_req, few_shot_example, gemini_client)` → `DamageAssessment`
- [ ] Calls `build_prompt_c()` from prompts.py
- [ ] Sends images + full prompt to `gemini-2.5-flash`
- [ ] Parses JSON into `DamageAssessment`
- [ ] Validates `claim_status`, `severity`, `issue_type` against enums
- [ ] Validates `supporting_image_ids` are real image IDs from the claim (not hallucinated)
- [ ] On parse failure: retry once
- [ ] On retry failure: return `DamageAssessment.fallback()`
- [ ] `time.sleep(1)` after every API call

### Checkpoint ✅
```bash
python code/agent3_assessor.py
# Run against 6 sample claims covering:
#   supported (user_001), contradicted (user_005),
#   not_enough_information (user_006), high severity (user_007 or similar),
#   blurry but supported (user_003), multi-image claim (user_002)
# Expected: claim_status matches ground truth for all 6
#           supporting_image_ids are valid image filenames
```

---

## Phase 7 — Agent 4: Risk & Assembly
**Goal:** Agent 4 synthesizing all prior outputs into final ClaimOutput with correct justification.

### Tasks

**`code/agent4_assembler.py`**
- [ ] `get_history_flags(user_id, user_history)` → `list[RiskFlag]` — pure Python, no LLM:
  - `user_history_risk` if: `rejected_claim > 0` OR `history_flags` contains `user_history_risk`
  - `manual_review_required` if: `history_flags` contains `manual_review_required`
  - Empty list if user not found
- [ ] `run(claim_row, parsed_claim, image_validation, damage_assessment, user_history, gemini_client)` → `ClaimOutput`
- [ ] Calls `build_prompt_d()` from prompts.py
- [ ] Sends text-only request (no images — Agent 4 works from summaries)
- [ ] Parses JSON into `ClaimOutput`
- [ ] **Post-parse enforcement:** if returned `claim_status` ≠ `damage_assessment.claim_status`, overwrite it — Agent 4 cannot change the verdict
- [ ] Merges VLM flags (Agent 2 + Agent 3) + history flags → deduplicated final list
- [ ] On parse failure: retry once
- [ ] On retry failure: assemble fallback from prior agent outputs directly (no LLM)
- [ ] `time.sleep(1)` after every API call

### Checkpoint ✅
```bash
python code/agent4_assembler.py
# Test with: clean history user + supported claim
#            high-risk history user (user_005) + contradicted claim
#            user_031: history risk flagged but claim_status=supported (history must NOT change status)
# Expected: correct flags merged, claim_status unchanged from Agent 3 value
#           justification mentions specific image IDs
#           history note appended when risk flags present
```

---

## Phase 8 — Pipeline Integration
**Goal:** `pipeline.py` wires all 4 agents, runs end-to-end on sample claims.

### Tasks

**`code/pipeline.py`**
- [ ] `run_pipeline(claims_path, output_path, dataset_base_dir)` — main entry point
- [ ] Startup: load all data (Stage 0), build FAISS index (Stage 1) — once only
- [ ] Per claim loop:
  ```
  try:
      few_shot = Stage1.retrieve(claim)
      parsed   = Agent1.run(claim)
      validated = Agent2.run(images, claim)
      if validated.valid_image:
          assessed = Agent3.run(images, parsed, evidence_req, few_shot)
      else:
          assessed = DamageAssessment.fast_path_fallback()
      output = Agent4.run(claim, parsed, validated, assessed, history)
      rows.append(output)
  except Exception as e:
      log(e)
      rows.append(ClaimOutput.from_fallback(claim))
  ```
- [ ] Progress logging: `[1/44] Processing user_002...`
- [ ] Write `output.csv` on completion via `stage0_loader.write_output_csv()`

### Checkpoint ✅
```bash
python code/pipeline.py --input dataset/sample_claims.csv --output output_sample.csv
# Expected: 20 rows in output_sample.csv
#           no crashes, all rows present, correct column headers
#           runtime under 5 minutes
```

---

## Phase 9 — Evaluation
**Goal:** Metrics computed, gate checked, evaluation_report.md filled.

### Tasks

**`evaluation/evaluate.py`**
- [ ] Load `output_sample.csv` + `dataset/sample_claims.csv`
- [ ] Exact match accuracy: `claim_status`, `severity`, `issue_type`, `object_part`, `evidence_standard_met`, `valid_image`
- [ ] Set-based F1 for `risk_flags`
- [ ] Print summary table
- [ ] Write to `evaluation/evaluation_report.md`
- [ ] **Gate check:** print PASS/FAIL for `claim_status` ≥ 0.70
- [ ] If FAIL: print which claims were wrong to guide prompt revision

### Checkpoint ✅
```bash
python evaluation/evaluate.py
# Expected: metrics table printed, report written
#           PASS or FAIL gate printed clearly
# If FAIL: revise Agent 3 prompt in prompts.py, re-run Phase 8 checkpoint, re-run this
```

---

## Phase 10 — Test Run & Submission
**Goal:** Full test run, output.csv verified, submission bundle ready.

### Tasks
- [ ] Run full pipeline:
  ```bash
  python code/pipeline.py --input dataset/claims.csv --output output.csv
  ```
- [ ] Verify `output.csv`: 44 rows, 14 columns, no empty cells, correct headers
- [ ] Validate enums: `python -c "import pandas as pd; df=pd.read_csv('output.csv'); print(df['claim_status'].unique())"`
- [ ] Fill actual numbers into `evaluation/evaluation_report.md` from run logs
- [ ] Write `README.md` (max 20 lines: install, set key, run pipeline, run eval)
- [ ] Build submission zip:
  ```bash
  zip -r code.zip code/ evaluation/ ARCHITECTURE.md IMPLEMENTATION.md README.md requirements.txt
  # Verify no dataset/, no .env, no __pycache__ included
  ```

### Final Submission Checklist ✅
- [ ] `code.zip` — all code, prompts, evaluation folder, docs
- [ ] `output.csv` — 44 rows from `dataset/claims.csv`
- [ ] `log.txt` — this chat transcript exported

---

## Phase Summary

| Phase | What Gets Built | Est. Time |
|---|---|---|
| 0 | Env setup, smoke tests | 20 min |
| 1 | schemas.py — all models, enums, validators | 30 min |
| 2 | Data loading, FAISS RAG | 30 min |
| 3 | prompts.py — all 4 agent prompts | 45 min |
| 4 | Agent 1: Claim Parser | 30 min |
| 5 | Agent 2: Image Validator | 35 min |
| 6 | Agent 3: Damage Assessor | 45 min |
| 7 | Agent 4: Risk & Assembly | 35 min |
| 8 | Pipeline integration | 30 min |
| 9 | Evaluation + gate check | 30 min |
| 10 | Test run + submission prep | 20 min |
| **Total** | | **~5.5 hours** |

Buffer for prompt iteration and debugging: **~3 hours**
Total realistic timeline: **8–9 hours** of the 24-hour window.

---

## Prompt Revision Protocol (if Phase 9 gate fails)

If `claim_status` accuracy < 0.70 after Phase 9:

1. Print which claims failed: `evaluate.py` already does this
2. Look at the pattern — is it one object type? One failure mode?
3. Edit only `prompts.py` — specifically `build_prompt_c()` for Agent 3
4. Common fixes:
   - Add explicit example of `contradicted` vs `not_enough_information` distinction
   - Strengthen the LOCATE step instruction
   - Add negative examples ("Do NOT return supported if damage is minor")
5. Re-run Phase 8 checkpoint (sample pipeline run)
6. Re-run Phase 9 checkpoint (evaluate)
7. Repeat until gate passes — then proceed to Phase 10
