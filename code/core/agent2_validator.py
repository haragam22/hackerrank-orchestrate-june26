"""
code/core/agent2_validator.py
Agent 2: Image Validator — vision call, no damage assessment.

Takes submitted images and determines:
  - valid_image: whether the image set is usable for damage review
  - image_quality_flags: list of detected quality issues
  - per_image_notes: brief notes per image

Run directly to self-test:
    python code/core/agent2_validator.py
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

from schemas import ImageValidation, RiskFlag, IMAGE_QUALITY_FLAGS
from prompts import PROMPT_B_SYSTEM, RETRY_SUFFIX, build_prompt_b

logger = logging.getLogger(__name__)

MODEL_TEXT          = "qwen/qwen2.5-vl-72b-instruct"
SLEEP_BETWEEN_CALLS = 4.5    # seconds between API calls
SLEEP_ON_RATE_LIMIT = 45.0   # seconds to sleep on 429
SLEEP_ON_503        = 45.0   # seconds to sleep on 503 (server overload)


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run(
    images: list[tuple[str, str]],
    claim_object: str,
    client,  # type: openai.OpenAI
) -> ImageValidation:
    """
    Agent 2: assess whether the submitted images are usable for damage review.

    Args:
        images:       list of (image_id, PIL.Image) from stage0_loader.load_images()
        claim_object: 'car', 'laptop', or 'package'
        client:       authenticated Gemini client

    Returns:
        ImageValidation — always returns something; falls back gracefully on failure.
    """
    # Fast path: no images at all
    if not images:
        logger.warning("Agent2: no images provided — returning fast_path_invalid")
        return ImageValidation.fast_path_invalid("No images were found for this claim.")

    # Fast path: check for missing images (Robust path verification)
    import os
    valid_paths = []
    for img_id, path in images:
        if not os.path.exists(path):
            logger.warning(f"Agent2: missing image path detected — {path}")
            return ImageValidation.fast_path_invalid(f"Image {img_id} could not be found.")
        valid_paths.append(path)

    prompt_user = build_prompt_b(claim_object)

    config = {
        "system_instruction": PROMPT_B_SYSTEM,
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    # ── Attempt 1 ─────────────────────────────────────────────────
    raw = _call_model(client, prompt_user, valid_paths, config, attempt=1)
    if raw is not None:
        result = _parse(raw, attempt=1)
        if result is not None:
            return result

    # ── Attempt 2 (retry with explicit schema reminder) ────────────
    logger.warning("Agent2: attempt 1 failed — retrying with RETRY_SUFFIX")
    retry_prompt = prompt_user + RETRY_SUFFIX
    raw = _call_model(client, retry_prompt, valid_paths, config, attempt=2)
    if raw is not None:
        result = _parse(raw, attempt=2)
        if result is not None:
            return result

    # ── Fallback ───────────────────────────────────────────────────
    logger.error("Agent2: both attempts failed — returning ImageValidation.fallback()")
    return ImageValidation.fallback()


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
            logger.info(f"Agent2 attempt {attempt} raw response: {raw[:300]}")
            return raw

        except openai.RateLimitError as e:
            logger.warning(f"Agent2: rate limit (429) on attempt {attempt} — sleeping {SLEEP_ON_RATE_LIMIT}s")
            time.sleep(SLEEP_ON_RATE_LIMIT)
        except openai.APIError as e:
            err_str = str(e)
            if "503" in err_str or "unavailable" in err_str.lower():
                logger.warning(f"Agent2: server unavailable (503) on attempt {attempt} — sleeping {SLEEP_ON_503}s")
                time.sleep(SLEEP_ON_503)
            else:
                logger.error(f"Agent2: API error on attempt {attempt}: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                return None
        except Exception as e:
            logger.error(f"Agent2: unknown error on attempt {attempt}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            return None
    return None


def _parse(raw: str, attempt: int) -> ImageValidation | None:
    """
    Parse raw JSON string into ImageValidation.
    The ImageValidation validator already filters flags to IMAGE_QUALITY_FLAGS only.
    Returns None on failure.
    """
    try:
        data = json.loads(raw)

        # Ensure valid_image is a proper bool (model sometimes returns string)
        vi = data.get("valid_image", False)
        if isinstance(vi, str):
            data["valid_image"] = vi.lower() == "true"

        result = ImageValidation(**data)
        logger.info(
            f"Agent2 attempt {attempt} parsed: "
            f"valid_image={result.valid_image}, "
            f"flags={[f.value for f in result.image_quality_flags]}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Agent2: JSON decode error on attempt {attempt}: {e}")
        return None
    except (ValidationError, TypeError, KeyError) as e:
        logger.warning(f"Agent2: schema validation error on attempt {attempt}: {e}")
        return None


# ---------------------------------------------------------------------------
# Self-test — run directly: python code/core/agent2_validator.py
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

    # Load helpers via importlib
    _spec = importlib.util.spec_from_file_location(
        "stage0_loader", pathlib.Path(__file__).parent / "stage0_loader.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_sample_claims = _mod.load_sample_claims
    load_images        = _mod.load_images

    samples = load_sample_claims(str(DATASET / "sample_claims.csv"))
    sample_map = {s["user_id"]: s for s in samples}

    # Test cases from IMPLEMENTATION.md:
    # - 2 clean valid images
    # - 1 blurry quality issue (user_003)
    # - 1 valid_image=false (user_008 — non_original_image)
    TEST_CASES = [
        ("user_001", "car",     True,  []),                        # clean, 1 image
        ("user_009", "laptop",  True,  []),                        # clean, 1 image
        ("user_003", "car",     True,  ["blurry_image"]),          # quality issue
        ("user_008", "car",     False, ["non_original_image"]),    # valid_image=false
        ("user_032", "package", False, ["cropped_or_obstructed"]), # valid_image=false
    ]

    print("=" * 65)
    print("agent2_validator.py self-test")
    print(f"Running Agent 2 on {len(TEST_CASES)} sample claims")
    print("=" * 65)

    all_valid_correct = True
    for uid, claim_object, gt_valid, gt_flags_hint in TEST_CASES:
        claim = sample_map.get(uid)
        if not claim:
            print(f"\n[!] {uid} not found in sample_claims.csv — skipping")
            continue

        images = load_images(claim["image_paths"], str(DATASET))
        print(f"\n{'─'*65}")
        print(f"  {uid} | object={claim_object} | images={len(images)}")
        print(f"  Ground truth: valid_image={gt_valid}, expected_flag_hint={gt_flags_hint}")

        result = run(images, claim_object, client)

        actual_flags = [f.value for f in result.image_quality_flags]
        valid_match  = result.valid_image == gt_valid
        flag_ok      = not gt_flags_hint or any(
            any(hint in f for f in actual_flags)
            for hint in gt_flags_hint
        )

        status_valid = "✅" if valid_match else "❌"
        status_flag  = "✅" if flag_ok  else "⚠️ "

        print(f"  Agent output: valid_image={result.valid_image}, flags={actual_flags}")
        print(f"  {status_valid} valid_image match={valid_match}")
        print(f"  {status_flag} flag hint present={flag_ok}")
        print(f"  Notes: {result.per_image_notes[:120]}")

        if not valid_match:
            all_valid_correct = False

    print("\n" + "=" * 65)
    if all_valid_correct:
        print("agent2_validator.py self-test complete ✅")
        print("All valid_image verdicts matched ground truth.")
    else:
        print("agent2_validator.py self-test complete ⚠️")
        print("Some valid_image verdicts did NOT match — review prompt B.")
    print("=" * 65)
