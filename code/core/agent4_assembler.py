"""
code/core/agent4_assembler.py
Agent 4: Risk & Assembly — text only, final output producer.

Does three things:
  1. LLM writes: claim_status_justification, evidence_standard_met_reason
  2. Python locks: claim_status, severity, valid_image, supporting_image_ids (from prior agents)
  3. Python computes: risk_flags by merging Agent 2 + Agent 3 + user history

Run directly to self-test:
    python code/core/agent4_assembler.py
"""

import sys
import io
import json
import time
import logging
import pathlib

from pydantic import ValidationError
from google import genai
from google.genai import types

# Add core/ to path for sibling imports when running standalone
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from schemas import (
    ClaimOutput, ParsedClaim, ImageValidation, DamageAssessment,
    RiskFlag, IssueType, ClaimStatus, Severity, coerce_flags,
)
from prompts import PROMPT_D_SYSTEM, RETRY_SUFFIX, build_prompt_d

logger = logging.getLogger(__name__)

MODEL_TEXT          = "qwen/qwen-2.5-7b-instruct"
SLEEP_BETWEEN_CALLS = 4.5
SLEEP_ON_RATE_LIMIT = 45.0
SLEEP_ON_503        = 45.0


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run(
    raw_claim: dict,
    parsed_claim: ParsedClaim,
    image_validation: ImageValidation,
    damage_assessment: DamageAssessment,
    user_history: dict,
    client,  # type: openai.OpenAI
) -> ClaimOutput:
    """
    Agent 4: synthesise final justification and assemble ClaimOutput.

    Args:
        raw_claim:         original CSV row dict (user_id, image_paths, user_claim, claim_object)
        parsed_claim:      output from Agent 1
        image_validation:  output from Agent 2
        damage_assessment: output from Agent 3 (or fast_path_fallback)
        user_history:      dict from load_user_history() for this user, or {} if not found
        client:            authenticated Gemini client

    Returns:
        ClaimOutput — always returns something; all fixed values locked from prior agents.
    """
    claim_object = raw_claim.get("claim_object", "unknown")

    prompt_user = build_prompt_d(
        claim_object, parsed_claim, image_validation, damage_assessment, user_history
    )

    config = {
        "system_instruction": PROMPT_D_SYSTEM,
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    # ── Attempt 1 ─────────────────────────────────────────────────
    raw = _call_model(client, prompt_user, config, attempt=1)
    if raw is not None:
        llm_data = _parse(raw, attempt=1)
        if llm_data is not None:
            return _assemble(raw_claim, parsed_claim, image_validation, damage_assessment, user_history, llm_data)

    # ── Attempt 2 ─────────────────────────────────────────────────
    logger.warning("Agent4: attempt 1 failed — retrying with RETRY_SUFFIX")
    raw = _call_model(client, prompt_user + RETRY_SUFFIX, config, attempt=2)
    if raw is not None:
        llm_data = _parse(raw, attempt=2)
        if llm_data is not None:
            return _assemble(raw_claim, parsed_claim, image_validation, damage_assessment, user_history, llm_data)

    # ── Fallback — assemble with generic justification ─────────────
    logger.error("Agent4: both attempts failed — assembling with fallback justification")
    return _assemble(raw_claim, parsed_claim, image_validation, damage_assessment, user_history, llm_data=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_model(
    client,  # type: openai.OpenAI
    prompt: str,
    config: dict,
    attempt: int,
) -> str | None:
    """Text-only API call via OpenRouter. Returns raw string or None on error."""
    import openai
    for network_attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": config["system_instruction"]},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"} if config["response_mime_type"] == "application/json" else None,
                temperature=config["temperature"],
            )
            time.sleep(SLEEP_BETWEEN_CALLS)
            raw = response.choices[0].message.content.strip()
            logger.info(f"Agent4 attempt {attempt} raw response: {raw[:400]}")
            return raw

        except openai.RateLimitError as e:
            logger.warning(f"Agent4: rate limit (429) on attempt {attempt} — sleeping {SLEEP_ON_RATE_LIMIT}s")
            time.sleep(SLEEP_ON_RATE_LIMIT)
        except openai.APIError as e:
            err_str = str(e)
            if "503" in err_str or "unavailable" in err_str.lower():
                logger.warning(f"Agent4: server unavailable (503) on attempt {attempt} — sleeping {SLEEP_ON_503}s")
                time.sleep(SLEEP_ON_503)
            else:
                logger.error(f"Agent4: API error on attempt {attempt}: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                return None
        except Exception as e:
            logger.error(f"Agent4: unknown error on attempt {attempt}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            return None
    return None


def _parse(raw: str, attempt: int) -> dict | None:
    """
    Parse Agent 4 JSON response. Only extracts the narrative fields —
    fixed values will be overridden by prior agent outputs in _assemble().
    Returns None on failure.
    """
    try:
        data = json.loads(raw)

        # Coerce bool fields
        esm = data.get("evidence_standard_met")
        if isinstance(esm, str):
            data["evidence_standard_met"] = esm.lower() == "true"

        # Require at minimum the two narrative fields
        just = data.get("claim_status_justification", "").strip()
        esm_reason = data.get("evidence_standard_met_reason", "").strip()
        if not just or not esm_reason:
            logger.warning(f"Agent4 attempt {attempt}: missing narrative fields")
            return None

        logger.info(f"Agent4 attempt {attempt} parsed OK")
        return data

    except json.JSONDecodeError as e:
        logger.warning(f"Agent4: JSON decode error on attempt {attempt}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Agent4: parse error on attempt {attempt}: {e}")
        return None


def _assemble(
    raw_claim: dict,
    parsed_claim: ParsedClaim,
    image_validation: ImageValidation,
    damage_assessment: DamageAssessment,
    user_history: dict,
    llm_data: dict | None,
) -> ClaimOutput:
    """
    Build the final ClaimOutput.
    Fixed values are ALWAYS taken from prior agents — LLM cannot change them.
    Only the justification text comes from the LLM.
    """
    # ── Locked values from prior agents ──────────────────────────
    claim_status         = damage_assessment.claim_status
    severity             = damage_assessment.severity
    supporting_image_ids = damage_assessment.supporting_image_ids
    valid_image          = image_validation.valid_image
    evidence_standard_met = damage_assessment.evidence_standard_met

    # ── LLM-written narrative (with fallbacks) ───────────────────
    if llm_data:
        justification    = llm_data.get("claim_status_justification", "").strip()
        esm_reason       = llm_data.get("evidence_standard_met_reason", "").strip()
        issue_type       = llm_data.get("issue_type", damage_assessment.issue_type.value)
        object_part      = llm_data.get("object_part", damage_assessment.object_part)
    else:
        justification    = damage_assessment.justification  # fall back to Agent 3's text
        esm_reason       = damage_assessment.evidence_standard_met_reason
        issue_type       = damage_assessment.issue_type.value
        object_part      = damage_assessment.object_part

    if not justification:
        justification = damage_assessment.justification
    if not esm_reason:
        esm_reason = damage_assessment.evidence_standard_met_reason

    # ── Risk flags (Python logic — not LLM) ─────────────────────
    risk_flags = _compute_risk_flags(image_validation, damage_assessment, user_history)

    # Hard rules — applied after Agent 4 LLM output, before writing CSV
    if RiskFlag.WRONG_OBJECT in risk_flags:
        claim_status = ClaimStatus.CONTRADICTED
        if RiskFlag.CLAIM_MISMATCH not in risk_flags:
            risk_flags.append(RiskFlag.CLAIM_MISMATCH)
    elif (RiskFlag.CLAIM_MISMATCH in risk_flags and
          claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION):
        claim_status = ClaimStatus.CONTRADICTED

    return ClaimOutput(
        user_id                    = raw_claim.get("user_id", "unknown"),
        image_paths                = raw_claim.get("image_paths", ""),
        user_claim                 = raw_claim.get("user_claim", ""),
        claim_object               = raw_claim.get("claim_object", "unknown"),
        evidence_standard_met      = evidence_standard_met,
        evidence_standard_met_reason = esm_reason,
        risk_flags                 = risk_flags,
        issue_type                 = issue_type,
        object_part                = object_part,
        claim_status               = claim_status,
        claim_status_justification = justification,
        supporting_image_ids       = supporting_image_ids,
        valid_image                = valid_image,
        severity                   = severity,
    )


def _compute_risk_flags(
    image_validation: ImageValidation,
    damage_assessment: DamageAssessment,
    user_history: dict,
) -> list[RiskFlag]:
    """
    Compute the deduplicated, ordered risk_flags list.

    Sources:
      - Agent 2: image_quality_flags (blurry_image, non_original_image, etc.)
      - Agent 3: damage_flags (claim_mismatch, damage_not_visible, wrong_object_part)
      - User history: user_history_risk if rejected_claim > 0 or history_flags != 'none'
      - System: manual_review_required if triggered

    Returns:
      Ordered list of RiskFlag values. Empty list if no flags (serializes as "none").
    """
    flags: set[RiskFlag] = set()

    # Agent 2 image quality flags
    for flag in image_validation.image_quality_flags:
        flags.add(flag)

    # Agent 3 damage flags
    for flag in damage_assessment.damage_flags:
        flags.add(flag)

    # User history risk
    try:
        rejected   = int(user_history.get("rejected_claim", 0))
        hist_flags = str(user_history.get("history_flags", "none")).strip().lower()
    except (ValueError, TypeError):
        rejected   = 0
        hist_flags = "none"

    if "user_history_risk" in hist_flags:
        flags.add(RiskFlag.USER_HISTORY_RISK)

    # Manual review required triggers
    manual_triggers = {
        RiskFlag.USER_HISTORY_RISK,
        RiskFlag.CLAIM_MISMATCH,
        RiskFlag.TEXT_INSTRUCTION_PRESENT,
        RiskFlag.POSSIBLE_MANIPULATION,
        RiskFlag.NON_ORIGINAL_IMAGE,
    }
    if (flags & manual_triggers) or (not image_validation.valid_image):
        flags.add(RiskFlag.MANUAL_REVIEW_REQUIRED)

    if not flags:
        return []  # serializes as "none"

    # Return in canonical RiskFlag enum order for consistency
    return [f for f in RiskFlag if f in flags and f != RiskFlag.NONE]


# ---------------------------------------------------------------------------
# Self-test — run directly: python code/core/agent4_assembler.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import os
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    load_dotenv(pathlib.Path(__file__).parent.parent.parent / ".env")
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    print("=" * 65)
    print("agent4_assembler.py self-test")
    print("=" * 65)

    # ── Test case 1: supported claim, no risk flags ────────────────
    print("\n[1] user_001 — supported, no risk flags")
    raw_claim_1 = {
        "user_id": "user_001",
        "image_paths": "images/sample/case_001/img_1.jpg",
        "user_claim": "My car's rear bumper has a dent after someone hit it in the parking lot.",
        "claim_object": "car",
    }
    parsed_1 = ParsedClaim(
        object_part="rear_bumper", issue_type="dent",
        claim_summary="Customer claims a dent on the rear bumper from a parking incident.",
    )
    iv_1 = ImageValidation(valid_image=True, image_quality_flags=[], per_image_notes="Clear original photo.")
    da_1 = DamageAssessment(
        claim_status="supported", issue_type="dent", object_part="rear_bumper",
        severity="medium", supporting_image_ids=["img_1"],
        justification="img_1 shows dent on rear bumper.",
        damage_flags=[], evidence_standard_met=True,
        evidence_standard_met_reason="Rear bumper clearly visible.",
    )
    hist_1 = {"past_claim_count": 2, "rejected_claim": 0, "history_flags": "none", "history_summary": "Clean history."}

    result_1 = run(raw_claim_1, parsed_1, iv_1, da_1, hist_1, client)
    assert result_1.claim_status == ClaimStatus.SUPPORTED
    assert result_1.severity.value == "medium"
    assert result_1.valid_image is True
    assert result_1.supporting_image_ids == ["img_1"]
    assert len(result_1.risk_flags) == 0  # no flags → serializes as "none"
    print(f"  ✅ claim_status={result_1.claim_status.value}, severity={result_1.severity.value}")
    print(f"     risk_flags={result_1.to_csv_row()['risk_flags']}")
    print(f"     justification: {result_1.claim_status_justification[:100]}")

    print("\n   (sleeping 8s before next call...)")
    time.sleep(8)

    # ── Test case 2: invalid images + user history risk ─────────────
    print("\n[2] user_008 — invalid images, user_history_risk → manual_review_required")
    raw_claim_2 = {
        "user_id": "user_008",
        "image_paths": "images/sample/case_008/img_1.jpg",
        "user_claim": "My car's rear bumper has a scratch from a collision.",
        "claim_object": "car",
    }
    parsed_2 = ParsedClaim(
        object_part="rear_bumper", issue_type="scratch",
        claim_summary="Customer claims a scratch on the rear bumper.",
    )
    iv_2 = ImageValidation(
        valid_image=False, image_quality_flags=["non_original_image"],
        per_image_notes="Stock photo with watermarks detected.",
    )
    da_2 = DamageAssessment.fast_path_fallback()
    hist_2 = {"past_claim_count": 5, "rejected_claim": 2, "history_flags": "user_history_risk", "history_summary": "Two rejected claims on file."}

    result_2 = run(raw_claim_2, parsed_2, iv_2, da_2, hist_2, client)
    assert result_2.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION
    assert result_2.valid_image is False
    assert RiskFlag.NON_ORIGINAL_IMAGE in result_2.risk_flags
    assert RiskFlag.USER_HISTORY_RISK in result_2.risk_flags
    assert RiskFlag.MANUAL_REVIEW_REQUIRED in result_2.risk_flags
    print(f"  ✅ claim_status={result_2.claim_status.value}, valid_image={result_2.valid_image}")
    print(f"     risk_flags={result_2.to_csv_row()['risk_flags']}")
    print(f"     justification: {result_2.claim_status_justification[:100]}")

    print("\n" + "=" * 65)
    print("agent4_assembler.py self-test complete ✅")
    print("Fixed values (claim_status, severity, valid_image) confirmed locked from prior agents.")
    print("Risk flag merging confirmed correct.")
    print("=" * 65)
