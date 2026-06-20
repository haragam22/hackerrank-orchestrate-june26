"""
code/core/agent1_parser.py
Agent 1: Claim Parser — text-only, no images.

Reads the user support conversation and extracts:
  - object_part: which part of the claimed object is damaged
  - issue_type: what type of damage is described
  - claim_summary: one-sentence claim summary

Run directly to self-test:
    python code/core/agent1_parser.py
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

from schemas import ParsedClaim, resolve_object_part
from prompts import PROMPT_A_SYSTEM, RETRY_SUFFIX, build_prompt_a

logger = logging.getLogger(__name__)

MODEL_TEXT = "qwen/qwen-2.5-7b-instruct"
SLEEP_BETWEEN_CALLS = 4.5    # seconds between API calls
SLEEP_ON_RATE_LIMIT = 45.0   # seconds to sleep on 429
SLEEP_ON_503        = 45.0   # seconds to sleep on 503 (server overload)


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run(
    user_claim: str,
    claim_object: str,
    client,  # type: openai.OpenAI
) -> ParsedClaim:
    """
    Agent 1: parse the user_claim conversation and extract structured fields.

    Args:
        user_claim:   raw conversation transcript string from claims.csv
        claim_object: 'car', 'laptop', or 'package'
        client:       authenticated Gemini client

    Returns:
        ParsedClaim — always returns something; falls back gracefully on failure.
    """
    prompt_user = build_prompt_a(user_claim, claim_object)

    config = {
        "system_instruction": PROMPT_A_SYSTEM,
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    # ── Attempt 1 ─────────────────────────────────────────────────
    raw = _call_model(client, prompt_user, config, attempt=1)
    if raw is not None:
        result = _parse(raw, claim_object, attempt=1)
        if result is not None:
            return result

    # ── Attempt 2 (retry with explicit schema reminder) ────────────
    logger.warning("Agent1: attempt 1 failed — retrying with RETRY_SUFFIX")
    retry_prompt = prompt_user + RETRY_SUFFIX
    raw = _call_model(client, retry_prompt, config, attempt=2)
    if raw is not None:
        result = _parse(raw, claim_object, attempt=2)
        if result is not None:
            return result

    # ── Fallback ───────────────────────────────────────────────────
    logger.error("Agent1: both attempts failed — returning ParsedClaim.fallback()")
    return ParsedClaim.fallback()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_model(
    client,  # type: openai.OpenAI
    prompt: str,
    config: dict,
    attempt: int,
) -> str | None:
    """
    Make one API call using OpenRouter (OpenAI-compatible client) and return raw text.
    Returns None on error (caller decides what to do).
    """
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
            logger.info(f"Agent1 attempt {attempt} raw response: {raw[:300]}")
            return raw

        except openai.RateLimitError as e:
            logger.warning(f"Agent1: rate limit (429) on attempt {attempt} — sleeping {SLEEP_ON_RATE_LIMIT}s")
            time.sleep(SLEEP_ON_RATE_LIMIT)
        except openai.APIError as e:
            err_str = str(e)
            if "503" in err_str or "unavailable" in err_str.lower():
                logger.warning(f"Agent1: server unavailable (503) on attempt {attempt} — sleeping {SLEEP_ON_503}s")
                time.sleep(SLEEP_ON_503)
            else:
                logger.error(f"Agent1: API error on attempt {attempt}: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                return None
        except Exception as e:
            logger.error(f"Agent1: unknown error on attempt {attempt}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            return None
    return None


def _parse(raw: str, claim_object: str, attempt: int) -> ParsedClaim | None:
    """
    Parse a raw JSON string into a ParsedClaim.
    Validates object_part against the claim_object enum.
    Returns None on failure.
    """
    try:
        data = json.loads(raw)

        # Coerce and validate object_part against the correct enum
        raw_part = data.get("object_part", "unknown")
        data["object_part"] = resolve_object_part(claim_object, str(raw_part))

        parsed = ParsedClaim(**data)
        logger.info(
            f"Agent1 attempt {attempt} parsed: "
            f"object_part={parsed.object_part}, "
            f"issue_type={parsed.issue_type.value}"
        )
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"Agent1: JSON decode error on attempt {attempt}: {e}")
        return None
    except (ValidationError, TypeError, KeyError) as e:
        logger.warning(f"Agent1: schema validation error on attempt {attempt}: {e}")
        return None


# ---------------------------------------------------------------------------
# Self-test — run directly: python code/core/agent1_parser.py
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

    client = genai.Client(api_key=API_KEY)

    # Load sample claims via importlib (avoid 'code' module conflict)
    BASE    = pathlib.Path(__file__).parent.parent.parent
    DATASET = BASE / "dataset"

    _spec = importlib.util.spec_from_file_location(
        "stage0_loader", pathlib.Path(__file__).parent / "stage0_loader.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_sample_claims = _mod.load_sample_claims

    samples = load_sample_claims(str(DATASET / "sample_claims.csv"))

    # Pick 5 test cases: mix of car, laptop, package
    TEST_USER_IDS = ["user_001", "user_002", "user_009", "user_019", "user_033"]
    test_cases = [s for s in samples if s["user_id"] in TEST_USER_IDS]

    # Fallback: just take first 5 if specific IDs not found
    if len(test_cases) < 3:
        test_cases = samples[:5]

    print("=" * 65)
    print("agent1_parser.py self-test")
    print(f"Running Agent 1 on {len(test_cases)} sample claims")
    print("=" * 65)

    all_passed = True
    for i, claim in enumerate(test_cases, 1):
        uid          = claim["user_id"]
        claim_object = claim["claim_object"]
        user_claim   = claim["user_claim"]
        gt_part      = claim.get("object_part", "?")
        gt_issue     = claim.get("issue_type", "?")

        print(f"\n[{i}] {uid} | object={claim_object}")
        print(f"     Ground truth: object_part={gt_part}, issue_type={gt_issue}")

        result = run(user_claim, claim_object, client)

        print(f"     Agent output: object_part={result.object_part}, issue_type={result.issue_type.value}")
        print(f"     Summary:      {result.claim_summary[:100]}")

        # Check that output is not a fallback
        is_fallback = result.object_part == "unknown" and result.issue_type.value == "unknown"
        if is_fallback:
            print(f"     ⚠️  FALLBACK returned — model may have failed")
            all_passed = False
        else:
            match_part  = result.object_part == gt_part
            match_issue = result.issue_type.value == gt_issue
            status = "✅" if (match_part and match_issue) else "⚠️ "
            print(f"     {status} part_match={match_part}, issue_match={match_issue}")

    print("\n" + "=" * 65)
    if all_passed:
        print("agent1_parser.py self-test complete ✅")
    else:
        print("agent1_parser.py self-test complete ⚠️  (some claims used fallback — check logs)")
    print("=" * 65)
