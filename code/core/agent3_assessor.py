"""
code/core/agent3_assessor.py
Agent 3: Damage Assessor — vision call, owns claim_status.

Given that images are valid (Agent 2 confirmed), assesses whether the
claimed damage is present and matches the claim.

Owns: claim_status, severity, supporting_image_ids, damage_flags

Run directly to self-test:
    python code/core/agent3_assessor.py
"""

import sys
import io
import json
import time
import logging
import pathlib

from pydantic import ValidationError
from PIL import Image as PILImage
from google import genai
from google.genai import types

# Add core/ to path for sibling imports when running standalone
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from schemas import (
    DamageAssessment, ParsedClaim, ImageValidation,
    ClaimStatus, Severity, IssueType,
    resolve_object_part, DAMAGE_FLAGS,
)
from prompts import PROMPT_C_SYSTEM, RETRY_SUFFIX, build_prompt_c

logger = logging.getLogger(__name__)

MODEL_TEXT          = "qwen/qwen2.5-vl-72b-instruct"
SLEEP_BETWEEN_CALLS = 4.5
SLEEP_ON_RATE_LIMIT = 45.0
SLEEP_ON_503        = 45.0


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run(
    images: list[tuple[str, str]],
    claim_object: str,
    parsed_claim: ParsedClaim,
    image_validation: ImageValidation,
    evidence_req: str,
    few_shot_example: dict | None,
    client,  # type: openai.OpenAI
) -> DamageAssessment:
    """
    Agent 3: assess whether the claimed damage is present in the images.

    IMPORTANT: This agent now always runs, regardless of valid_image status.

    Args:
        images:           list of (image_id, PIL.Image) from stage0_loader.load_images()
        claim_object:     'car', 'laptop', or 'package'
        parsed_claim:     output from Agent 1
        image_validation: output from Agent 2
        evidence_req:     evidence requirement text from stage0_loader.get_evidence_requirement()
        few_shot_example: dict from stage1_rag.retrieve_example() or None
        client:           authenticated Gemini client

    Returns:
        DamageAssessment — always returns something; claim_status is locked here.
    """
    if not images:
        logger.warning("Agent3: no images provided — returning fast_path_fallback")
        return DamageAssessment.fast_path_fallback()

    image_ids  = [img_id for img_id, _ in images]  # valid IDs for hallucination check

    # Fast path verification (though Agent 2 should have caught it)
    import os
    valid_paths = []
    for _, path in images:
        if not os.path.exists(path):
            return DamageAssessment.fast_path_fallback()
        valid_paths.append(path)

    prompt_user = build_prompt_c(
        claim_object, parsed_claim, image_validation, evidence_req, few_shot_example, image_ids
    )

    config = {
        "system_instruction": PROMPT_C_SYSTEM,
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    # ── Attempt 1 ─────────────────────────────────────────────────
    raw = _call_model(client, prompt_user, valid_paths, config, attempt=1)
    if raw is not None:
        result = _parse(raw, claim_object, image_ids, attempt=1)
        if result is not None:
            return result

    # ── Attempt 2 ─────────────────────────────────────────────────
    logger.warning("Agent3: attempt 1 failed — retrying with RETRY_SUFFIX")
    retry_prompt = prompt_user + RETRY_SUFFIX
    raw = _call_model(client, retry_prompt, valid_paths, config, attempt=2)
    if raw is not None:
        result = _parse(raw, claim_object, image_ids, attempt=2)
        if result is not None:
            return result

    # ── Fallback ──────────────────────────────────────────────────
    logger.error("Agent3: both attempts failed — returning DamageAssessment.fallback()")
    return DamageAssessment.fallback()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str) -> str:
    import base64
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _call_model(
    client,  # type: openai.OpenAI
    prompt: str,
    image_paths: list[str],
    config: dict,
    attempt: int,
) -> str | None:
    """
    Make one vision API call: text prompt + base64 images via OpenRouter.
    Returns raw JSON string or None on error.
    """
    import openai
    for network_attempt in range(5):
        try:
            content_payload = [{"type": "text", "text": prompt}]
            for path in image_paths:
                b64_img = _encode_image(path)
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })

            response = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": config["system_instruction"]},
                    {"role": "user", "content": content_payload}
                ],
                response_format={"type": "json_object"} if config["response_mime_type"] == "application/json" else None,
                temperature=config["temperature"],
            )
            time.sleep(SLEEP_BETWEEN_CALLS)
            raw = response.choices[0].message.content.strip()
            logger.info(f"Agent3 attempt {attempt} raw response: {raw[:300]}")
            return raw

        except openai.RateLimitError as e:
            logger.warning(f"Agent3: rate limit (429) on attempt {attempt} — sleeping {SLEEP_ON_RATE_LIMIT}s")
            time.sleep(SLEEP_ON_RATE_LIMIT)
        except openai.APIError as e:
            err_str = str(e)
            if "503" in err_str or "unavailable" in err_str.lower():
                logger.warning(f"Agent3: server unavailable (503) on attempt {attempt} — sleeping {SLEEP_ON_503}s")
                time.sleep(SLEEP_ON_503)
            else:
                logger.error(f"Agent3: API error on attempt {attempt}: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                return None
        except Exception as e:
            logger.error(f"Agent3: unknown error on attempt {attempt}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            return None
    return None


def _parse(
    raw: str,
    claim_object: str,
    valid_image_ids: list[str],
    attempt: int,
) -> DamageAssessment | None:
    """
    Parse raw JSON into DamageAssessment.
    - Validates object_part against claim_object enum.
    - Validates supporting_image_ids are real IDs (not hallucinated filenames).
    Returns None on failure.
    """
    try:
        data = json.loads(raw)

        # Coerce and validate object_part
        raw_part = data.get("object_part", "unknown")
        data["object_part"] = resolve_object_part(claim_object, str(raw_part))

        # Validate supporting_image_ids — keep only IDs that actually exist
        raw_ids = data.get("supporting_image_ids", [])
        if isinstance(raw_ids, list):
            validated_ids = []
            for raw_id in raw_ids:
                # Strip extension if present
                sid = str(raw_id).strip()
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    if sid.lower().endswith(ext):
                        sid = sid[: -len(ext)]
                # Only keep IDs that match real loaded images
                if sid in valid_image_ids:
                    validated_ids.append(sid)
                else:
                    logger.warning(f"Agent3: dropped hallucinated image_id '{raw_id}' (valid: {valid_image_ids})")
            data["supporting_image_ids"] = validated_ids
        else:
            data["supporting_image_ids"] = []

        # Coerce valid_image / evidence_standard_met bools (model may return strings)
        for bool_field in ("evidence_standard_met",):
            val = data.get(bool_field)
            if isinstance(val, str):
                data[bool_field] = val.lower() == "true"

        result = DamageAssessment(**data)
        logger.info(
            f"Agent3 attempt {attempt} parsed: "
            f"claim_status={result.claim_status.value}, "
            f"severity={result.severity.value}, "
            f"supporting_ids={result.supporting_image_ids}, "
            f"esm={result.evidence_standard_met}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Agent3: JSON decode error on attempt {attempt}: {e}")
        return None
    except (ValidationError, TypeError, KeyError) as e:
        logger.warning(f"Agent3: schema validation error on attempt {attempt}: {e}")
        return None


# ---------------------------------------------------------------------------
# Self-test — run directly: python code/core/agent3_assessor.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import os
    import importlib.util
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    load_dotenv(pathlib.Path(__file__).parent.parent.parent / ".env")
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    client  = genai.Client(api_key=API_KEY)
    BASE    = pathlib.Path(__file__).parent.parent.parent
    DATASET = BASE / "dataset"

    # Load helpers
    def _load_module(name):
        spec = importlib.util.spec_from_file_location(
            name, pathlib.Path(__file__).parent / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    stage0 = _load_module("stage0_loader")
    samples = stage0.load_sample_claims(str(DATASET / "sample_claims.csv"))
    ev_reqs = stage0.load_evidence_requirements(str(DATASET / "evidence_requirements.csv"))
    sample_map = {s["user_id"]: s for s in samples}

    # Test cases: 3 diverse cases spread out to avoid rate limits
    # (supported, contradicted, not_enough_information)
    TEST_CASES = [
        ("user_001", "car",  "supported",              "medium"),   # clean dent, 1 image
        ("user_005", "car",  "contradicted",            "none"),     # mismatch claim
        ("user_006", "car",  "not_enough_information",  "unknown"),  # wrong angle
    ]
    INTER_TEST_SLEEP = 8.0  # seconds between test cases to avoid rate limit

    print("=" * 65)
    print("agent3_assessor.py self-test")
    print(f"Running Agent 3 on {len(TEST_CASES)} sample claims")
    print("=" * 65)

    status_correct = 0
    total = 0

    for uid, claim_object, gt_status, gt_severity in TEST_CASES:
        claim = sample_map.get(uid)
        if not claim:
            print(f"\n[!] {uid} not found in sample_claims — skipping")
            continue

        images = stage0.load_images(claim["image_paths"], str(DATASET))
        evidence_req = stage0.get_evidence_requirement(claim_object, ev_reqs)

        # Build ParsedClaim from ground truth (Agent 1 already tested separately)
        parsed = ParsedClaim(
            object_part=claim.get("object_part", "unknown"),
            issue_type=claim.get("issue_type", "unknown"),
            claim_summary=f"Customer claims {claim.get('issue_type','damage')} on {claim.get('object_part','part')}.",
        )

        print(f"\n{'─'*65}")
        print(f"  {uid} | object={claim_object} | images={len(images)}")
        print(f"  GT: claim_status={gt_status}, severity={gt_severity}")
        print(f"  Claimed: {parsed.object_part} / {parsed.issue_type.value}")

        result = run(images, claim_object, parsed, evidence_req, None, client)

        status_match = result.claim_status.value == gt_status
        is_fallback    = result.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION and not result.evidence_standard_met_reason.startswith("Assessment")

        status_icon = "✅" if status_match else "❌"
        total += 1
        if status_match:
            status_correct += 1

        print(f"  Agent: claim_status={result.claim_status.value}, severity={result.severity.value}")
        print(f"  {status_icon} status_match={status_match}")
        print(f"  supporting_ids={result.supporting_image_ids}")
        print(f"  justification: {result.justification[:120]}")
        if result.damage_flags:
            print(f"  damage_flags: {[f.value for f in result.damage_flags]}")

        # Wait between test cases to avoid rate limit cascades
        if total < len(TEST_CASES):
            print(f"  (sleeping {INTER_TEST_SLEEP}s before next call...)")
            time.sleep(INTER_TEST_SLEEP)

    accuracy = status_correct / total if total > 0 else 0
    print("\n" + "=" * 65)
    print(f"agent3_assessor.py self-test complete")
    print(f"claim_status accuracy: {status_correct}/{total} = {accuracy:.0%}")
    gate = "✅ GATE PASSED (≥70%)" if accuracy >= 0.70 else "❌ GATE FAILED (<70%) — revise Prompt C"
    print(gate)
    print("=" * 65)
