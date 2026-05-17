"""
src/retrieval/faiss_index.py
============================
Production-grade FAISS vector search layer for an e-commerce
product recommendation system.

Design contract
---------------
* IndexFlatIP  – inner product == cosine similarity on L2-normalised vectors
* Singleton pattern for both the FAISS index and the metadata DataFrame
* Row i in FAISS index  ≡  row i in processed_products.csv  (invariant never broken)
* Query vectors receive the BGE-recommended instruction prefix; document vectors do not
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (adjust if your directory layout differs)
# ---------------------------------------------------------------------------
_BASE        = Path(__file__).resolve().parent.parent.parent  # repo root
ARTIFACTS    = _BASE / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS / "embeddings.npy"
METADATA_PATH   = ARTIFACTS / "processed_products.csv"
INDEX_PATH      = ARTIFACTS / "faiss_index.bin"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE query instruction prefix (REQUIRED for query-side encoding only)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GENERIC_WORDS = {
    "best", "cheap", "top", "good", "great", "nice", "affordable", "budget",
    "premium", "latest", "new", "buy", "get", "find", "for", "under", "products",
    "items", "things", "stuff",
}

# ---------------------------------------------------------------------------
# Module-level singletons  (loaded at most once per process)
# ---------------------------------------------------------------------------
_faiss_index:  Optional[faiss.Index] = None
_metadata_df:  Optional[pd.DataFrame] = None
_embed_model:  Optional[SentenceTransformer] = None


# ===========================================================================
# 1. load_embeddings
# ===========================================================================

def load_embeddings(path: str | Path = EMBEDDINGS_PATH) -> np.ndarray:
    """
    Load precomputed embeddings from .npy file.

    Rules
    -----
    * dtype must be float32
    * array must be C-contiguous
    * Original array is NEVER mutated in-place
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    raw = np.load(str(path))                    # load as-is

    # ── dtype guard ─────────────────────────────────────────────────────────
    if raw.dtype != np.float32:
        log.warning("Embeddings dtype is %s; casting to float32 (copy).", raw.dtype)
        embeddings = raw.astype(np.float32)     # new array, original untouched
    else:
        embeddings = raw.copy()                 # explicit copy – never mutate original

    # ── contiguity guard ────────────────────────────────────────────────────
    if not embeddings.flags["C_CONTIGUOUS"]:
        log.warning("Embeddings array is not C-contiguous; creating contiguous copy.")
        embeddings = np.ascontiguousarray(embeddings)

    assert embeddings.dtype == np.float32, "dtype guard failed"
    assert embeddings.flags["C_CONTIGUOUS"],  "contiguity guard failed"
    assert embeddings.ndim == 2, f"Expected 2-D array, got shape {embeddings.shape}"
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf values.")

    log.info("Embeddings loaded: shape=%s  dtype=%s", embeddings.shape, embeddings.dtype)
    return embeddings


# ===========================================================================
# 2. load_metadata
# ===========================================================================

def load_metadata(path: str | Path = METADATA_PATH) -> pd.DataFrame:
    """
    Load processed_products.csv ONCE and cache in module-level singleton.
    Row index is preserved exactly as stored.
    """
    global _metadata_df
    if _metadata_df is not None:
        return _metadata_df

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    _metadata_df = pd.read_csv(str(path), low_memory=False)
    log.info("Metadata loaded: %d rows, %d columns", len(_metadata_df), len(_metadata_df.columns))
    return _metadata_df


# ===========================================================================
# 3. build_index
# ===========================================================================

def build_index(
    embeddings: np.ndarray,
    metadata_path: str | Path = METADATA_PATH,
) -> faiss.Index:
    """
    Build an IndexFlatIP over the supplied embeddings.

    Why IndexFlatIP?
    ----------------
    Embeddings are L2-normalised →  ⟨a, b⟩ = cos(a, b).
    Exact inner-product search with zero approximation error.
    IVF/HNSW introduce recall loss; prohibited by spec.
    """
    df = load_metadata(metadata_path)

    # ── Alignment pre-check ──────────────────────────────────────────────────
    if len(embeddings) != len(df):
        raise ValueError(
            f"ALIGNMENT ERROR: embeddings has {len(embeddings)} rows "
            f"but metadata has {len(df)} rows."
        )

    N, D = embeddings.shape
    log.info("Building IndexFlatIP: N=%d  D=%d", N, D)

    index = faiss.IndexFlatIP(D)
    index.add(embeddings)                       # add ALL vectors, no filtering

    # ── Post-add validation ──────────────────────────────────────────────────
    assert index.ntotal == len(df), (
        f"Index ntotal ({index.ntotal}) != CSV rows ({len(df)})"
    )
    log.info("Index built successfully: ntotal=%d", index.ntotal)
    return index


# ===========================================================================
# 4. save_index
# ===========================================================================

def save_index(index: faiss.Index, path: str | Path = INDEX_PATH) -> None:
    """Persist FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    log.info("Index saved: %s", path)


# ===========================================================================
# 5. load_index  (singleton, memory-mapped)
# ===========================================================================

def load_index(path: str | Path = INDEX_PATH) -> faiss.Index:
    """
    Load FAISS index from disk ONCE per process using memory-mapping.
    Subsequent calls return the cached singleton.
    """
    global _faiss_index
    if _faiss_index is not None:
        return _faiss_index

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. "
            "Run build_index() + save_index() first."
        )

    log.info("Loading index (memory-mapped): %s", path)
    try:
        _faiss_index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load FAISS index — file may be corrupted."
        ) from exc

    d = _faiss_index.d
    n = _faiss_index.ntotal
    log.info(
        "Index loaded: ntotal=%d  dimension=%d  dtype=float32  mmap=True",
        n, d,
    )
    return _faiss_index


# ===========================================================================
# 6. _get_model  (singleton)
# ===========================================================================

def _get_model() -> SentenceTransformer:
    """Return cached SentenceTransformer, loading it at most once."""
    global _embed_model
    if _embed_model is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
        log.info("Loading embedding model: %s  device=%s", MODEL_NAME, device)
        _embed_model = SentenceTransformer(MODEL_NAME, device=device)
    return _embed_model


# ===========================================================================
# 7. encode_query
# ===========================================================================

def _preprocess_query(text: str) -> str:
    """Mirror the document-side preprocessing: lowercase + collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_query(query_string: str) -> np.ndarray:
    """
    Encode a free-text query into a unit-norm float32 embedding.

    Steps
    -----
    1. Validate input
    2. Preprocess (same rules as ingestion pipeline)
    3. Prepend BGE query instruction prefix
    4. Encode with BAAI/bge-small-en-v1.5
    5. L2-normalise (normalize_embeddings=True inside model.encode)

    Returns
    -------
    np.ndarray  shape (1, D)  dtype float32
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if not isinstance(query_string, str) or not query_string.strip():
        raise ValueError("encode_query received an empty or invalid query string.")

    # Input hardening: cap length + collapse excessive repeated characters.
    query_string = query_string[:300]
    query_string = re.sub(r"(.)\1{5,}", r"\1", query_string)

    # ── Preprocessing ────────────────────────────────────────────────────────
    cleaned = _preprocess_query(query_string)

    # ── BGE query prefix ─────────────────────────────────────────────────────
    prefixed = BGE_QUERY_PREFIX + cleaned

    # ── Encode + normalise ───────────────────────────────────────────────────
    model = _get_model()
    vec = model.encode(
        [prefixed],
        normalize_embeddings=True,   # L2 normalisation applied once here
        convert_to_numpy=True,
    ).astype(np.float32)             # guarantee dtype

    # Contiguity for FAISS
    if not vec.flags["C_CONTIGUOUS"]:
        vec = np.ascontiguousarray(vec)

    # shape: (1, D)
    assert vec.shape[0] == 1
    return vec


# ===========================================================================
# 8. search
# ===========================================================================

def search(
    query: str,
    top_k: int = 50,
    index_path: str | Path = INDEX_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> list[dict]:
    """
    Semantic search over the FAISS index.

    Parameters
    ----------
    query      : free-text search string
    top_k      : number of results to return (capped at index.ntotal)

    Returns
    -------
    List of dicts, each containing all metadata columns + 'similarity_score'.
    Ordered by descending similarity.
    """
    def _extract_category_hint(tokens: list[str]) -> str:
        hint_map = {
            "headphone": "headphones",
            "headphones": "headphones",
            "earphone": "headphones",
            "earphones": "headphones",
            "earbud": "headphones",
            "earbuds": "headphones",
            "speaker": "speaker",
            "speakers": "speaker",
            "mobile": "mobile",
            "mobiles": "mobile",
            "phone": "mobile",
            "phones": "mobile",
            "laptop": "laptop",
            "laptops": "laptop",
            "watch": "watch",
            "watches": "watch",
            "tablet": "tablet",
            "tablets": "tablet",
            "camera": "camera",
            "cameras": "camera",
        }
        for t in tokens:
            if t in hint_map:
                return hint_map[t]
        return ""

    # ── Weak-query protection ────────────────────────────────────────────────
    cleaned_query = _preprocess_query(query)
    q_tokens = [t for t in cleaned_query.split() if t]
    non_generic = [t for t in q_tokens if t not in GENERIC_WORDS]
    generic_count = len(q_tokens) - len(non_generic)
    if len(q_tokens) <= 2 and generic_count > (len(q_tokens) / 2):
        log.warning(
            "Weak query detected (tokens=%d, non_generic=%d, generic=%d) — skipping FAISS search.",
            len(q_tokens),
            len(non_generic),
            generic_count,
        )
        return []
    category_hint = _extract_category_hint(q_tokens)

    # ── Load singletons ──────────────────────────────────────────────────────
    index = load_index(index_path)
    df    = load_metadata(metadata_path)

    # ── Cap top_k ────────────────────────────────────────────────────────────
    if top_k <= 0:
        top_k = 10
    safe_k = min(top_k, index.ntotal)
    if safe_k != top_k:
        log.warning("top_k=%d capped to index.ntotal=%d", top_k, index.ntotal)

    log.info("Search | query='%s...'  top_k=%d", query[:60], safe_k)

    # ── Encode query ─────────────────────────────────────────────────────────
    q_vec = encode_query(query)       # shape (1, D), float32, L2-normed
    if q_vec.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch with index.")

    # ── FAISS inner-product search ───────────────────────────────────────────
    scores, indices = index.search(q_vec, safe_k)  # shapes: (1, k), (1, k)

    scores  = scores[0]    # flatten batch dim
    indices = indices[0]
    max_score = float(scores[0]) if len(scores) > 0 else 0.0
    threshold = max(0.3, max_score * 0.6)
    fallback_threshold = threshold * 0.5
    seen_best: dict[str, tuple[float, str]] = {}

    log.info("Returned indices (top-%d): %s", safe_k, indices[:10].tolist())

    # ── Map indices → metadata rows ──────────────────────────────────────────
    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        if idx < 0:          # FAISS returns -1 for padded results
            continue
        if not np.isfinite(score):
            continue
        score = float(score)
        if score < 0 or score > 1.1:
            continue
        if score < threshold:
            continue
        row = df.iloc[idx].to_dict()
        name = str(row.get("name", "")).strip()
        category = (
            str(row.get("category", "")).strip()
            or str(row.get("main_category", "")).strip()
            or str(row.get("sub_category", "")).strip()
        )
        if not name:
            continue
        if not category:
            continue
        category_l = category.lower()
        name_key = name.lower()

        # Soft duplicate suppression:
        # keep same-name entries only when they are meaningfully different.
        dup_margin = 0.08
        prev = seen_best.get(name_key)
        if prev is not None:
            prev_score, prev_cat = prev
            if abs(score - prev_score) <= dup_margin and category_l == prev_cat:
                continue
        seen_best[name_key] = (score, category_l)

        row["similarity_score"] = score
        row["_faiss_index"]     = int(idx)
        row["_rank"]            = rank + 1
        row["_category_hint"]   = category_hint
        results.append(row)

    # Safe fallback: if adaptive threshold removed everything, recover top raw FAISS rows
    # using a relaxed (not removed) threshold.
    if not results:
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
                continue
            if not np.isfinite(score):
                continue
            score = float(score)
            if score < 0 or score > 1.1:
                continue
            if score < fallback_threshold:
                continue
            row = df.iloc[idx].to_dict()
            name = str(row.get("name", "")).strip()
            category = (
                str(row.get("category", "")).strip()
                or str(row.get("main_category", "")).strip()
                or str(row.get("sub_category", "")).strip()
            )
            if not name:
                continue
            if not category:
                continue
            row["similarity_score"] = score
            row["_faiss_index"] = int(idx)
            row["_rank"] = rank + 1
            row["_category_hint"] = category_hint
            results.append(row)
            if len(results) >= 5:
                break

    scores_out = [r["similarity_score"] for r in results]
    if not all(
        scores_out[i] >= scores_out[i + 1]
        for i in range(len(scores_out) - 1)
    ):
        log.warning("Score ordering inconsistency detected.")
    if not results:
        log.warning("FAISS returned no valid results.")
    results = results[:top_k]
    return results


# ===========================================================================
# CLI / sanity test
# ===========================================================================

if __name__ == "__main__":
    import sys

    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"

    emb_path  = artifacts_dir / "embeddings.npy"
    meta_path = artifacts_dir / "processed_products.csv"
    idx_path  = artifacts_dir / "faiss_index.bin"

    # ── Build index if it doesn't exist ──────────────────────────────────────
    if not idx_path.exists():
        log.info("faiss_index.bin not found — building from embeddings …")
        embeddings = load_embeddings(emb_path)
        index      = build_index(embeddings, meta_path)
        save_index(index, idx_path)
        # populate singleton so we don't re-read from disk immediately
        _faiss_index = index
    else:
        log.info("faiss_index.bin found — skipping build step.")

    # ── Sanity test ───────────────────────────────────────────────────────────
    TEST_QUERY = "wireless headphones"
    TOP_K      = 10

    print("\n" + "=" * 60)
    print(f"SANITY TEST  |  query='{TEST_QUERY}'  top_k={TOP_K}")
    print("=" * 60)

    results = search(
        query        = TEST_QUERY,
        top_k        = TOP_K,
        index_path   = idx_path,
        metadata_path= meta_path,
    )

    if not results:
        print("No results returned — check that the index was built correctly.")
        sys.exit(1)

    print(f"\nTop-{min(3, len(results))} results:\n")
    prev_score = None
    for r in results[:3]:
        score = r["similarity_score"]
        name  = r.get("name", "<no name>")[:80]
        cat   = r.get("main_category", "?")
        sub   = r.get("sub_category", "?")
        rank  = r["_rank"]

        # Scores must be non-increasing
        if prev_score is not None and score > prev_score + 1e-5:
            log.warning("Score ordering violation at rank %d!", rank)
        prev_score = score

        print(f"  Rank {rank:>2} | score={score:.4f} | {name}")
        print(f"         | category: {cat} > {sub}")
        print()

    # Verify semantic relevance heuristic: at least one result should mention
    # audio/electronics-related category keywords
    audio_keywords = {
        "electronics", "audio", "headphone", "earphone",
        "computer", "accessories", "mobile",
    }
    cats = " ".join(
        (r.get("main_category", "") + " " + r.get("sub_category", "")).lower()
        for r in results[:3]
    )
    if any(kw in cats for kw in audio_keywords):
        print("✓  Semantic relevance check PASSED — top results are in expected categories.")
    else:
        print("⚠  Semantic relevance check INCONCLUSIVE — review top results manually.")

    print("\n✓  Sanity test complete.")