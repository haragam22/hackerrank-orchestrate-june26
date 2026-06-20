"""
code/stage0_loader.py
Stage 0: Data loading — CSVs, images, and output writing.
Pure Python, no LLM calls. Run once at pipeline startup.

Run directly to verify:
    python code/stage0_loader.py
"""

import sys
import io
import logging
import pathlib
from typing import Optional

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV Loaders
# ---------------------------------------------------------------------------

def load_claims(path: str) -> list[dict]:
    """
    Load claims.csv (or any claims file) into a list of row dicts.
    Columns expected: user_id, image_paths, user_claim, claim_object
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"user_id", "image_paths", "user_claim", "claim_object"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"load_claims: missing columns {missing} in {path}")
    records = df.to_dict(orient="records")
    logger.info(f"load_claims: loaded {len(records)} rows from {path}")
    return records


def load_user_history(path: str) -> dict[str, dict]:
    """
    Load user_history.csv into a dict keyed by user_id.
    Columns: user_id, past_claim_count, accept_claim, manual_review_claim,
             rejected_claim, last_90_days_claim_count, history_flags, history_summary
    Returns empty dict entry if user not found (handled by caller).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"user_id", "past_claim_count", "rejected_claim", "history_flags", "history_summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"load_user_history: missing columns {missing} in {path}")

    history = {}
    for row in df.to_dict(orient="records"):
        uid = row["user_id"]
        # Cast numeric fields safely
        row["past_claim_count"] = _safe_int(row.get("past_claim_count", 0))
        row["accept_claim"] = _safe_int(row.get("accept_claim", 0))
        row["manual_review_claim"] = _safe_int(row.get("manual_review_claim", 0))
        row["rejected_claim"] = _safe_int(row.get("rejected_claim", 0))
        row["last_90_days_claim_count"] = _safe_int(row.get("last_90_days_claim_count", 0))
        history[uid] = row

    logger.info(f"load_user_history: loaded {len(history)} user records from {path}")
    return history


def load_evidence_requirements(path: str) -> dict[tuple[str, str], str]:
    """
    Load evidence_requirements.csv into a dict keyed by (claim_object, applies_to).
    Also includes ('all', applies_to) entries that apply to all objects.

    Returns:
        dict mapping (claim_object, applies_to) -> minimum_image_evidence text
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"claim_object", "applies_to", "minimum_image_evidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"load_evidence_requirements: missing columns {missing} in {path}")

    reqs = {}
    for row in df.to_dict(orient="records"):
        key = (row["claim_object"].strip().lower(), row["applies_to"].strip().lower())
        reqs[key] = row["minimum_image_evidence"].strip()

    logger.info(f"load_evidence_requirements: loaded {len(reqs)} requirement entries from {path}")
    return reqs


def load_sample_claims(path: str) -> list[dict]:
    """
    Load sample_claims.csv (all 14 labeled output columns).
    Used for FAISS index building and evaluation.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    records = df.to_dict(orient="records")
    logger.info(f"load_sample_claims: loaded {len(records)} labeled samples from {path}")
    return records


def load_images(image_paths_str: str, base_dir: str) -> list[tuple[str, str]]:
    """
    Parse a semicolon-separated image_paths string, resolve against base_dir,
    and return a list of (image_id, absolute_path) tuples.

    image_id is the filename without extension (e.g. 'img_1').

    NOTE: image paths in the CSVs are relative to the dataset/ directory.
    base_dir should be the absolute path to the dataset/ folder.
    Example: base_dir = '/path/to/repo/dataset'

    Missing files are still appended (with their absolute path) so downstream
    agents can explicitly handle the failure for robust path verification.
    """
    import os
    base = pathlib.Path(base_dir)
    results = []

    if not image_paths_str or not image_paths_str.strip():
        logger.warning("load_images: empty image_paths string")
        return results

    for raw_path in image_paths_str.split(";"):
        raw_path = raw_path.strip()
        if not raw_path:
            continue

        full_path = base / raw_path
        abs_path = os.path.abspath(str(full_path))
        image_id = full_path.stem  # filename without extension
        results.append((image_id, abs_path))
        logger.debug(f"load_images: resolved '{raw_path}' to '{abs_path}'")

    if not results:
        logger.warning(f"load_images: no valid images found in '{image_paths_str}'")

    return results


# ---------------------------------------------------------------------------
# Evidence requirement lookup
# ---------------------------------------------------------------------------

def get_evidence_requirement(
    claim_object: str,
    evidence_reqs: dict[tuple[str, str], str],
) -> str:
    """
    Return a combined evidence requirement text for the given claim_object.
    Gathers all 'all' entries + object-specific entries and joins them.
    Returns a fallback string if nothing found.
    """
    obj = claim_object.strip().lower()
    lines = []

    for (req_obj, applies_to), text in evidence_reqs.items():
        if req_obj in (obj, "all"):
            lines.append(f"- [{applies_to}]: {text}")

    if not lines:
        return "The submitted images should clearly show the claimed object and damage."

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output_csv(rows: list[dict], output_path: str) -> None:
    """
    Write a list of to_csv_row() dicts to output_path.
    Ensures column order matches OUTPUT_COLUMNS from schemas.
    """
    OUTPUT_COLUMNS = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]

    if not rows:
        logger.warning("write_output_csv: no rows to write")
        return

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"write_output_csv: wrote {len(rows)} rows to {output_path}")
    print(f"Output written to {output_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default: int = 0) -> int:
    """Parse a value to int safely, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    BASE = pathlib.Path(__file__).parent.parent.parent  # repo root (code/core/ → code/ → root)
    DATASET = BASE / "dataset"

    print("=" * 60)
    print("stage0_loader.py self-test")
    print("=" * 60)

    # ── 1. load_claims ────────────────────────────────────────────
    print("\n[1] load_claims (claims.csv — 44 test rows)")
    claims = load_claims(str(DATASET / "claims.csv"))
    assert len(claims) == 44, f"Expected 44 claims, got {len(claims)}"
    assert "user_id" in claims[0]
    assert "image_paths" in claims[0]
    assert "claim_object" in claims[0]
    print(f"   ✅ {len(claims)} claims loaded")
    print(f"      Sample: {claims[0]['user_id']} | {claims[0]['claim_object']}")

    # ── 2. load_sample_claims ─────────────────────────────────────
    print("\n[2] load_sample_claims (sample_claims.csv — 20 labeled rows)")
    samples = load_sample_claims(str(DATASET / "sample_claims.csv"))
    assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"
    assert "claim_status" in samples[0]
    assert "valid_image" in samples[0]
    print(f"   ✅ {len(samples)} labeled samples loaded")
    print(f"      Columns: {list(samples[0].keys())[:5]}...")

    # ── 3. load_user_history ──────────────────────────────────────
    print("\n[3] load_user_history (user_history.csv)")
    history = load_user_history(str(DATASET / "user_history.csv"))
    assert len(history) >= 1
    assert "user_001" in history
    u = history["user_001"]
    assert isinstance(u["rejected_claim"], int)
    assert isinstance(u["past_claim_count"], int)
    print(f"   ✅ {len(history)} user records loaded")
    print(f"      user_001: past_claim_count={u['past_claim_count']}, rejected={u['rejected_claim']}, flags={u['history_flags']}")

    # ── 4. load_evidence_requirements ────────────────────────────
    print("\n[4] load_evidence_requirements (evidence_requirements.csv)")
    ev_reqs = load_evidence_requirements(str(DATASET / "evidence_requirements.csv"))
    assert len(ev_reqs) >= 1
    print(f"   ✅ {len(ev_reqs)} requirement entries loaded")
    # Show one entry
    for k, v in list(ev_reqs.items())[:1]:
        print(f"      Sample key={k}: {v[:60]}...")

    # ── 5. get_evidence_requirement ───────────────────────────────
    print("\n[5] get_evidence_requirement lookup")
    car_req = get_evidence_requirement("car", ev_reqs)
    laptop_req = get_evidence_requirement("laptop", ev_reqs)
    package_req = get_evidence_requirement("package", ev_reqs)
    assert len(car_req) > 10
    assert len(laptop_req) > 10
    assert len(package_req) > 10
    print(f"   ✅ car requirement: {len(car_req)} chars")
    print(f"   ✅ laptop requirement: {len(laptop_req)} chars")
    print(f"   ✅ package requirement: {len(package_req)} chars")

    # ── 6. load_images ────────────────────────────────────────────
    # Note: image paths in CSVs are relative to dataset/, so base_dir = DATASET
    print("\n[6] load_images (sample case_001)")
    first_sample = samples[0]
    images = load_images(first_sample["image_paths"], str(DATASET))
    assert len(images) >= 1, "Expected at least 1 image for case_001"
    for img_id, img in images:
        assert img.mode == "RGB"
        print(f"   ✅ Loaded '{img_id}' — size={img.size}")

    # ── 7. load_images — missing file graceful skip ───────────────
    print("\n[7] load_images — graceful handling of missing files")
    bad_images = load_images("images/sample/case_001/img_1.jpg;nonexistent/img_99.jpg", str(DATASET))
    assert len(bad_images) == 1, "Should load 1 valid + skip 1 missing"
    print(f"   ✅ 1 valid image loaded, missing file skipped gracefully")

    print("\n" + "=" * 60)
    print("stage0_loader.py self-test complete ✅  All 7 checks passed.")
    print("=" * 60)
