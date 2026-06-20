"""
code/stage1_rag.py
Stage 1: FAISS-backed RAG retrieval using Gemini embeddings.
Embeds all sample_claims user_claim texts once at startup,
then retrieves the most similar example for each test claim.

Run directly to verify:
    python code/stage1_rag.py
"""

import sys
import io
import time
import logging
from typing import Optional

import numpy as np
import faiss
from google import genai

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
SIMILARITY_THRESHOLD = 0.75   # below this → zero-shot (no few-shot injection)
SLEEP_BETWEEN_CALLS = 4.5     # seconds between embedding API calls


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_text(text: str, client: genai.Client) -> np.ndarray:
    """
    Embed a single text string using Gemini embedding model.
    Returns a float32 numpy array.
    Raises on API error (caller handles retries).
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Index build — run once at pipeline startup
# ---------------------------------------------------------------------------

def build_faiss_index(
    sample_claims: list[dict],
    client: genai.Client,
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed all sample claim texts and build an in-memory FAISS index.

    Args:
        sample_claims: list of dicts from load_sample_claims()
        client: authenticated Gemini client

    Returns:
        (index_matrix, sample_claims) where:
          - index_matrix: np.ndarray of shape (N, embed_dim), L2-normalised
          - sample_claims: original list (passed through for retrieval later)
    """
    logger.info(f"build_faiss_index: embedding {len(sample_claims)} sample claims...")
    vectors = []

    for i, claim in enumerate(sample_claims):
        text = claim.get("user_claim", "").strip()
        if not text:
            logger.warning(f"build_faiss_index: empty user_claim at index {i}, using placeholder")
            text = "unknown claim"

        vec = _embed_text(text, client)
        # L2-normalise so inner product == cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)

        if (i + 1) % 5 == 0:
            logger.info(f"  Embedded {i + 1}/{len(sample_claims)}...")

        time.sleep(SLEEP_BETWEEN_CALLS)

    index_matrix = np.vstack(vectors).astype(np.float32)
    logger.info(
        f"build_faiss_index: done — matrix shape={index_matrix.shape}, "
        f"embed_dim={index_matrix.shape[1]}"
    )
    return index_matrix, sample_claims


def build_faiss_index_cached(
    sample_claims: list[dict],
    client: genai.Client,
    cache_path,
) -> np.ndarray:
    """
    Build (or load from disk cache) the FAISS embedding matrix.

    If a valid .npz cache exists at cache_path with the same number of samples,
    it is loaded instantly (zero API calls). Otherwise, embeds sample_claims
    and saves the result to cache_path for future runs.

    Args:
        sample_claims: list of dicts from load_sample_claims()
        client:        authenticated Gemini client
        cache_path:    pathlib.Path where cache is stored (e.g. rag_cache.npz)

    Returns:
        index_matrix: np.ndarray of shape (N, embed_dim), L2-normalised
    """
    import pathlib
    cache_path = pathlib.Path(cache_path)

    # ── Try to load from cache ────────────────────────────────────
    if cache_path.exists():
        try:
            data = np.load(str(cache_path))
            cached_matrix = data["matrix"]
            cached_n = int(data["n_samples"])
            if cached_n == len(sample_claims):
                logger.info(
                    f"build_faiss_index_cached: loaded from cache "
                    f"({cache_path.name}, shape={cached_matrix.shape}) — 0 API calls"
                )
                return cached_matrix
            else:
                logger.warning(
                    f"build_faiss_index_cached: cache size mismatch "
                    f"({cached_n} vs {len(sample_claims)}) — re-embedding"
                )
        except Exception as e:
            logger.warning(f"build_faiss_index_cached: cache load failed ({e}) — re-embedding")

    # ── Cache miss: embed and save ────────────────────────────────
    logger.info(f"build_faiss_index_cached: cache not found, embedding {len(sample_claims)} samples...")
    index_matrix, _ = build_faiss_index(sample_claims, client)

    try:
        np.savez(str(cache_path), matrix=index_matrix, n_samples=len(sample_claims))
        logger.info(f"build_faiss_index_cached: cache saved to {cache_path.name}")
    except Exception as e:
        logger.warning(f"build_faiss_index_cached: could not save cache — {e}")

    return index_matrix


# ---------------------------------------------------------------------------
# Retrieval — called per claim during pipeline run
# ---------------------------------------------------------------------------

def retrieve_example(
    query_claim: str,
    index_matrix: np.ndarray,
    sample_claims: list[dict],
    client: genai.Client,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[dict]:
    """
    Embed the query claim text and find the most similar sample.

    Args:
        query_claim: raw user_claim text from the current test claim
        index_matrix: L2-normalised embedding matrix from build_faiss_index()
        sample_claims: original sample_claims list (same order as index_matrix)
        client: authenticated Gemini client
        threshold: minimum cosine similarity to return a result (default 0.75)

    Returns:
        The most similar sample_claims dict if similarity >= threshold, else None.
        Logs the similarity score every time.
    """
    query_vec = _embed_text(query_claim.strip(), client)

    # L2-normalise query
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Cosine similarities (inner product of normalised vectors)
    sims = index_matrix @ query_vec  # shape (N,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    best_sample = sample_claims[best_idx]
    logger.info(
        f"retrieve_example: best match = {best_sample.get('user_id', '?')} "
        f"| sim={best_sim:.4f} | threshold={threshold}"
    )

    time.sleep(SLEEP_BETWEEN_CALLS)

    if best_sim >= threshold:
        logger.info(f"retrieve_example: RETRIEVED (sim={best_sim:.4f} >= {threshold})")
        return best_sample
    else:
        logger.info(f"retrieve_example: ZERO-SHOT (sim={best_sim:.4f} < {threshold})")
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import os
    import pathlib
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    BASE = pathlib.Path(__file__).parent.parent.parent  # repo root (code/core/ → code/ → root)
    DATASET = BASE / "dataset"

    # Import loader via importlib (avoid 'code' shadowing Python built-in)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "stage0_loader", pathlib.Path(__file__).parent / "stage0_loader.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_sample_claims = _mod.load_sample_claims

    print("=" * 60)
    print("stage1_rag.py self-test")
    print("=" * 60)

    # ── Load samples ──────────────────────────────────────────────
    print("\n[1] Loading sample claims for index build...")
    samples = load_sample_claims(str(DATASET / "sample_claims.csv"))
    print(f"   Loaded {len(samples)} samples")

    # ── Build index ───────────────────────────────────────────────
    print("\n[2] Building FAISS index (embeds 20 sample claims)...")
    print("    This makes 20 API calls — ~25 seconds...")
    index_matrix, indexed_samples = build_faiss_index(samples, client)
    assert index_matrix.shape[0] == 20, f"Expected 20 rows, got {index_matrix.shape[0]}"
    print(f"   ✅ Index built — shape={index_matrix.shape}")

    # ── Test queries ──────────────────────────────────────────────
    print("\n[3] Running 3 test retrieval queries...")

    test_queries = [
        (
            "My car has a dent on the rear bumper from a parking accident.",
            "car dent rear_bumper — should retrieve a similar car claim"
        ),
        (
            "The laptop screen has a crack running from top to bottom after it fell.",
            "laptop screen crack — should retrieve or return None"
        ),
        (
            "Package arrived and the box is completely crushed on one side.",
            "package crushed — should retrieve or return None"
        ),
    ]

    for query_text, description in test_queries:
        print(f"\n   Query: {description}")
        print(f"   Text:  {query_text[:70]}...")
        result = retrieve_example(query_text, index_matrix, indexed_samples, client)
        if result:
            print(f"   → Retrieved: {result['user_id']} | status={result.get('claim_status')} | object={result.get('claim_object')}")
        else:
            print(f"   → Zero-shot (no match above threshold)")

    print("\n" + "=" * 60)
    print("stage1_rag.py self-test complete ✅")
    print("=" * 60)
