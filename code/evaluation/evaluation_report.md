# Evaluation Report
## Multi-Modal Evidence Review — Vision-RAG Orchestrator

---

## 1. System Overview

**Model used:** `gemini-2.5-flash` (Google AI Studio free tier)
**Embedding model:** `gemini-embedding-001` (Google AI Studio free tier)
**Pipeline:** 4-stage sequential (RAG retrieval → text extraction → vision assessment → assembly)
**Run date:** _(fill after run)_
**Total runtime:** _(fill after run)_

---

## 2. Sample Set Accuracy (dataset/sample_claims.csv — 20 rows)

### Per-Field Exact Match Accuracy

| Field | Correct | Total | Accuracy |
|---|---|---|---|
| `claim_status` | ___ | 20 | ___% |
| `severity` | ___ | 20 | ___% |
| `issue_type` | ___ | 20 | ___% |
| `object_part` | ___ | 20 | ___% |
| `evidence_standard_met` | ___ | 20 | ___% |
| `valid_image` | ___ | 20 | ___% |

### Risk Flags Set-Based F1

| Metric | Value |
|---|---|
| Precision | ___% |
| Recall | ___% |
| F1 | ___% |

### Evaluation Gate
- `claim_status` accuracy ≥ 0.70 required before test run
- **Result:** _(PASSED / FAILED — fill after run)_

---

## 3. Model Calls & Token Usage

### Sample Run (20 claims)

| Stage | API Calls | Approx Input Tokens | Approx Output Tokens |
|---|---|---|---|
| Stage 1 — Embeddings (index build) | 20 | ~10,000 | 0 |
| Stage 1 — Embeddings (queries) | 20 | ~2,000 | 0 |
| Stage 2 — Text extraction | 20 | ~15,000 | ~3,000 |
| Stage 3 — Vision assessment | 20 | ~40,000 | ~8,000 |
| Retries (if any) | ___ | ___ | ___ |
| **Total** | **~80** | **~67,000** | **~11,000** |

### Test Run (44 claims)

| Stage | API Calls | Approx Input Tokens | Approx Output Tokens |
|---|---|---|---|
| Stage 1 — Embeddings (index build) | 20 | ~10,000 | 0 |
| Stage 1 — Embeddings (queries) | 44 | ~4,400 | 0 |
| Stage 2 — Text extraction | 44 | ~33,000 | ~6,600 |
| Stage 3 — Vision assessment | 44 | ~88,000 | ~17,600 |
| Retries (if any) | ___ | ___ | ___ |
| **Total** | **~152** | **~135,000** | **~24,200** |

_Note: Token counts are estimates. Actual counts logged per call during run._

---

## 4. Images Processed

| Run | Claims | Images | Avg per Claim |
|---|---|---|---|
| Sample | 20 | ~38 | ~1.9 |
| Test | 44 | ~82 | ~1.86 |
| **Total** | **64** | **~120** | — |

---

## 5. Cost Estimate

**Pricing assumption:** Google AI Studio free tier — $0.00 for all calls within rate limits.

| Run | VLM Cost | Embedding Cost | Total |
|---|---|---|---|
| Sample run | $0.00 | $0.00 | $0.00 |
| Test run | $0.00 | $0.00 | $0.00 |
| **Total** | **$0.00** | **$0.00** | **$0.00** |

**If free tier exhausted (paid tier fallback pricing for gemini-2.5-flash):**
- Input: $0.30/1M tokens × ~135K tokens = ~$0.04
- Output: $2.50/1M tokens × ~24K tokens = ~$0.06
- **Worst case paid cost: ~$0.10 total**

---

## 6. Latency & Runtime

| Metric | Value |
|---|---|
| Avg time per claim (Stage 3 vision call) | _(fill after run)_ sec |
| Total sample run time | _(fill after run)_ min |
| Total test run time | _(fill after run)_ min |
| Sleep between calls | 1 second |

---

## 7. Rate Limit Strategy

**Model:** `gemini-2.5-flash` free tier
**Limit:** 15 requests per minute (RPM), no hard daily cap

**Strategy:**
- Sequential processing with `time.sleep(1)` after every API call
- At ~3–5 seconds per claim (API latency + sleep), 44 claims complete in ~3–4 minutes
- Peak RPM never exceeds ~12 — well within the 15 RPM limit
- No batching needed at this scale

**Retry policy:**
- 1 retry on JSON parse failure
- Exponential backoff not needed at this scale — flat 1s sleep is sufficient
- If rate limit error received (429): sleep 10s and retry once

**Caching:**
- FAISS index built once at pipeline start from 20 sample embeddings — not rebuilt per claim
- User history and evidence requirements loaded once into memory dicts — O(1) lookup per claim

---

## 8. Failure Analysis

| Failure Type | Count | Handling |
|---|---|---|
| JSON parse failures (first attempt) | ___ | Retried with schema reminder |
| JSON parse failures (after retry) | ___ | Fallback row written |
| Missing image files | ___ | Logged, skipped, `valid_image=false` |
| RAG retrievals below threshold | ___ | Proceeded zero-shot |
| Claims with no history record | ___ | Treated as new user, no history flags |

---

## 9. Observations & Known Limitations

_(Fill after run — note any patterns in errors, prompt issues, or model behavior)_

- 
- 
- 

---

## 10. Prompt Iteration Log

_(Document any prompt changes made between sample eval and test run)_

| Iteration | Change Made | Impact on sample accuracy |
|---|---|---|
| v1 | Initial prompt | claim_status accuracy: ___% |
| v2 | _(if changed)_ | claim_status accuracy: ___% |
