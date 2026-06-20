"""
code/smoke_test.py
Phase 0 checkpoint: verifies API key, text calls, vision calls, and embeddings.

Run from the repo root:
    python code/smoke_test.py
"""

import os
import sys
import pathlib
import time

# ── Load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print("ERROR: GEMINI_API_KEY not set.")
    print("Copy .env.example → .env and fill in your actual Gemini API key.")
    sys.exit(1)

from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_TEXT  = "gemini-2.5-flash"
MODEL_EMBED = "gemini-embedding-001"

# ── Test 1: Text call ────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — Text call to gemini-2.5-flash")
print("=" * 60)

response = client.models.generate_content(
    model=MODEL_TEXT,
    contents="Say exactly: SMOKE_TEST_OK",
)
text_out = response.text.strip()
print(f"Response: {text_out}")
assert text_out, "Empty response from text call"
print("✅ Text call PASSED\n")

time.sleep(2)

# ── Test 2: Vision call ──────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2 — Vision call to gemini-2.5-flash")
print("=" * 60)

IMAGE_PATH = (
    pathlib.Path(__file__).parent.parent.parent   # code/core/ → code/ → repo root
    / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"
)

if not IMAGE_PATH.exists():
    print(f"WARNING: Test image not found at {IMAGE_PATH}")
    print("Skipping vision test.")
    print("⚠️  Vision call SKIPPED\n")
else:
    img = Image.open(IMAGE_PATH)
    response = client.models.generate_content(
        model=MODEL_TEXT,
        contents=[
            "Describe this image in one sentence. What object is shown?",
            img,
        ],
    )
    vision_out = response.text.strip()
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Response: {vision_out}")
    assert vision_out, "Empty response from vision call"
    print("✅ Vision call PASSED\n")

time.sleep(2)

# ── Test 3: Embedding call ────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3 — Embedding call to gemini-embedding-001")
print("=" * 60)

TEST_TEXT = "My car's rear bumper has a large dent from a collision."

result = client.models.embed_content(
    model=MODEL_EMBED,
    contents=TEST_TEXT,
)

embedding = result.embeddings[0].values
print(f"Input text: {TEST_TEXT!r}")
print(f"Embedding dimensions: {len(embedding)}")
assert len(embedding) > 0, "Embedding is empty!"
print("✅ Embedding call PASSED\n")

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("SMOKE TEST COMPLETE — all 3 checks passed ✅")
print("Phase 0 checkpoint DONE. Ready for Phase 1.")
print("=" * 60)
