"""
production_pipeline.py
======================
Deterministic data ingestion and embedding pipeline for an
e-commerce product recommendation system.

Pipeline stages
---------------
1.  Load raw CSV
2.  Clean & normalise columns
3.  Extract brand (hybrid deterministic strategy)
4.  Handle missing descriptions
5.  Deduplicate (deterministic)
6.  Build embedding_text
7.  Truncate to token-safe length
8.  Enforce stable ordering
9.  Generate L2-normalised embeddings  (BAAI/bge-small-en-v1.5)
10. Validate alignment
11. Save processed_products.csv + embeddings.npy
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
# Constants
# ---------------------------------------------------------------------------
INPUT_FILE        = "final_dataset_v2.csv"
OUTPUT_CSV        = "processed_products.csv"
OUTPUT_NPY        = "embeddings.npy"
MODEL_NAME        = "BAAI/bge-small-en-v1.5"
BATCH_SIZE        = 64
MAX_EMB_CHARS     = 2048          # ~512 tokens safety cap

GENERIC_TOKENS    = frozenset({
    "the", "a", "an", "new", "best", "top", "buy", "get",
    "for", "with", "and", "or", "in", "of", "by", "set",
})

OUTPUT_COLUMNS = [
    "name", "main_category", "sub_category",
    "final_price", "ratings", "no_of_ratings",
    "brand", "image", "link", "embedding_text",
]

# ---------------------------------------------------------------------------
# 1. Helpers – text normalisation
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Lowercase, unicode-normalise, collapse whitespace, strip."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"[^\w\s\-.,!?&%()/:'+]", " ", text)  # keep semantic chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_category(value) -> str:
    """Lowercase + strip for category columns."""
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _clean_price(value) -> Optional[int]:
    """
    Remove ₹, commas, whitespace; convert to int.
    Returns None if conversion fails or price <= 0.
    """
    try:
        cleaned = re.sub(r"[₹,\s]", "", str(value))
        price = int(float(cleaned))
        return price if price > 0 else None
    except (ValueError, TypeError):
        return None


def _clean_ratings(series: pd.Series) -> pd.Series:
    """Convert to float32, clip [0, 5], fill NaN with median."""
    num = pd.to_numeric(series, errors="coerce").astype("float32")
    median = float(np.nanmedian(num.values))
    num = num.fillna(median).clip(0.0, 5.0)
    return num


def _clean_no_of_ratings(series: pd.Series) -> pd.Series:
    """Convert to int64, fill NaN with mode."""
    num = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    mode_val = num.mode()
    fill_val = int(mode_val.iloc[0]) if not mode_val.empty else 0
    return num.fillna(fill_val).astype("int64")


# ---------------------------------------------------------------------------
# 2. Brand extraction
# ---------------------------------------------------------------------------

def _extract_brand_from_name(name: str) -> Optional[str]:
    """Step 1: first token of cleaned product name, minus punctuation."""
    token = name.strip().split()[0] if name.strip() else ""
    token = re.sub(r"[^\w]", "", token).lower()
    if token and token not in GENERIC_TOKENS and len(token) > 1:
        return token
    return None


def _extract_brand_from_text(text: str) -> Optional[str]:
    """Step 2: first meaningful capitalised token from product_text (original case)."""
    if not isinstance(text, str) or not text.strip():
        return None
    for tok in text.split():
        clean = re.sub(r"[^\w]", "", tok)
        if (
            clean
            and clean[0].isupper()
            and clean.lower() not in GENERIC_TOKENS
            and len(clean) > 1
        ):
            return clean.lower()
    return None


def extract_brand(row: pd.Series) -> str:
    """Hybrid deterministic brand extraction."""
    brand = _extract_brand_from_name(str(row.get("name", "")))
    if brand:
        return brand
    brand = _extract_brand_from_text(row.get("product_text", ""))
    return brand if brand else "unknown"


# ---------------------------------------------------------------------------
# 3. Data cleaning function
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning rules.  Returns cleaned copy; logs drop reasons.
    """
    initial = len(df)
    log.info("Total rows loaded: %d", initial)

    # --- Drop missing name ---
    mask_name = df["name"].isna() | (df["name"].astype(str).str.strip() == "")
    drop_name = mask_name.sum()
    df = df[~mask_name].copy()

    # --- Clean price ---
    df["final_price"] = df["final_price"].apply(_clean_price)
    mask_price = df["final_price"].isna()
    drop_price = mask_price.sum()
    df = df[~mask_price].copy()
    df["final_price"] = df["final_price"].astype("int64")

    log.info("Rows dropped – missing/empty name  : %d", drop_name)
    log.info("Rows dropped – invalid/missing price: %d", drop_price)

    # --- Ratings ---
    df["ratings"]        = _clean_ratings(df["ratings"])
    df["no_of_ratings"]  = _clean_no_of_ratings(df["no_of_ratings"])

    # --- Categories ---
    df["main_category"] = df["main_category"].apply(_clean_category)
    df["sub_category"]  = df["sub_category"].apply(_clean_category)

    # --- Text fields ---
    df["name"] = df["name"].apply(lambda x: _normalise_text(str(x)))

    # --- product_text: missing → replace with name ---
    def fix_product_text(row):
        pt = row.get("product_text", "")
        if pd.isna(pt) or str(pt).strip() == "":
            return str(row["name"])
        return _normalise_text(str(pt))

    df["product_text"] = df.apply(fix_product_text, axis=1)

    log.info("Rows after cleaning: %d  (dropped %d total)",
             len(df), initial - len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Normalisation & brand column
# ---------------------------------------------------------------------------

def normalise_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Extract brand; keep original index stable."""
    log.info("Extracting brand column …")
    df = df.copy()
    df["brand"] = df.apply(extract_brand, axis=1)
    return df


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by name → sort by no_of_ratings DESC → keep first.
    Deterministic because sort is stable.
    """
    before = len(df)
    df = (
        df.sort_values("no_of_ratings", ascending=False, kind="stable")
          .drop_duplicates(subset=["name"], keep="first")
    )
    log.info("Rows after deduplication: %d  (removed %d duplicates)",
             len(df), before - len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Embedding text builder
# ---------------------------------------------------------------------------

def build_embedding_text(row: pd.Series) -> str:
    """
    Construct embedding_text from validated fields.
    Format is EXACT per spec.
    """
    text = (
        f"Product Name: {row['name']}\n"
        f"Category: {row['main_category']} > {row['sub_category']}\n"
        f"Price: {row['final_price']}\n"
        f"Rating: {row['ratings']:.1f} ({row['no_of_ratings']} reviews)\n"
        f"Brand: {row['brand']}\n"
        f"Description: {row['product_text']}"
    )
    return re.sub(r"\n{2,}", "\n", text).strip()


def truncate_embedding_text(text: str, max_chars: int = MAX_EMB_CHARS) -> str:
    """
    Truncate to max_chars while preserving priority fields:
    1. Product Name line
    2. Category line
    3. Everything else (description is last → truncated first)
    """
    if len(text) <= max_chars:
        return text

    lines   = text.split("\n")
    # Priority lines (first 5: Name, Category, Price, Rating, Brand)
    priority = lines[:5]
    desc_line = lines[5] if len(lines) > 5 else ""

    base      = "\n".join(priority)
    remaining = max_chars - len(base) - 1       # -1 for "\n"

    if remaining > len("Description: ") + 5:
        desc_truncated = desc_line[:remaining]
        return base + "\n" + desc_truncated
    return base


def build_all_embedding_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply builder + truncation; log samples."""
    df = df.copy()
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)
    df["embedding_text"] = df["embedding_text"].apply(truncate_embedding_text)
    return df


# ---------------------------------------------------------------------------
# 7. Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
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
    log.info("Device selected: %s", device.upper())
    return device


# ---------------------------------------------------------------------------
# 8. Embedding generator
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str], device: str) -> np.ndarray:
    """
    Encode texts with BAAI/bge-small-en-v1.5.
    Returns L2-normalised float32 array of shape (N, dim).
    """
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device=device)

    log.info("Generating embeddings for %d texts (batch_size=%d) …",
             len(texts), BATCH_SIZE)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalisation built-in
        convert_to_numpy=True,
    )

    # Explicit L2 norm guard (ensures unit vectors regardless of model version)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = (embeddings / norms).astype(np.float32)

    log.info("Embedding shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# 9. Alignment validation
# ---------------------------------------------------------------------------

def validate_alignment(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(df) != len(embeddings):
        raise ValueError(
            f"ALIGNMENT MISMATCH: DataFrame has {len(df)} rows "
            f"but embeddings has {len(embeddings)} vectors. "
            "Pipeline halted."
        )
    log.info("Alignment validated: %d rows == %d vectors ✓", len(df), len(embeddings))


# ---------------------------------------------------------------------------
# 10. Logging helpers
# ---------------------------------------------------------------------------

def log_samples(df: pd.DataFrame) -> None:
    samples = pd.concat([df.head(2), df.tail(2)])
    log.info("──── Sample embedding_text ────")
    for idx, row in samples.iterrows():
        preview = row["embedding_text"][:200].replace("\n", " | ")
        log.info("  [%d] %s …", idx, preview)
    log.info("───────────────────────────────")


# ---------------------------------------------------------------------------
# 11. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_file: str = INPUT_FILE) -> None:
    # ── Load ────────────────────────────────────────────────────────────────
    log.info("Loading: %s", input_file)
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        log.error("Input file not found: %s", input_file)
        sys.exit(1)

    log.info("Columns detected: %s", list(df.columns))

    # ── Clean ───────────────────────────────────────────────────────────────
    df = clean_dataframe(df)

    # ── Normalise & enrich ──────────────────────────────────────────────────
    df = normalise_and_enrich(df)

    # ── Deduplicate ─────────────────────────────────────────────────────────
    df = deduplicate(df)

    # ── Stable ordering (CRITICAL – no shuffle ever) ─────────────────────────
    df = df.reset_index(drop=True)

    # ── Build embedding texts ────────────────────────────────────────────────
    df = build_all_embedding_texts(df)

    # ── Log samples ──────────────────────────────────────────────────────────
    log_samples(df)

    # ── Detect device ────────────────────────────────────────────────────────
    device = detect_device()

    # ── Generate embeddings ──────────────────────────────────────────────────
    texts      = df["embedding_text"].tolist()
    embeddings = generate_embeddings(texts, device)

    # ── Validate alignment ───────────────────────────────────────────────────
    validate_alignment(df, embeddings)

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_df = df[OUTPUT_COLUMNS].copy()
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved: %s  (%d rows)", OUTPUT_CSV, len(out_df))

    np.save(OUTPUT_NPY, embeddings)
    log.info("Saved: %s  shape=%s dtype=%s",
             OUTPUT_NPY, embeddings.shape, embeddings.dtype)

    log.info("Pipeline complete. ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    run_pipeline(csv_path)