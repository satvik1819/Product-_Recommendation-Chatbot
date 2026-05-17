"""
src/conversation/responder.py

Deterministic Conversation & Explanation Layer
Sits ON TOP of the existing retrieval pipeline — never modifies it.

Hardened per spec (full production pass):
  • normalize_query applied to every effective_query (Parts 1.1)
  • SYNONYM EXPANSION via expand_tokens + SYNONYMS dict (Part 1.2)
  • soft_match() replaces all strict 't in product_text' checks (Part 1.3)
  • is_low_signal_query guard before retrieval
  • Balanced adaptive threshold: 0.25 / 0.35 / 0.45 (Part 2.7)
  • Updated lexical score uses soft_match (Part 2.1)
  • Category boost via detect_category() (Part 3)
  • Relaxed intent penalty: 0.4 → 0.6 (Part 2.3)
  • Multi-stage filtering: Pass1 (penalty) → Pass2 (no-penalty) → Pass3 (engine-only) (Part 2.4)
  • NEVER EMPTY RULE: fallback to original[:5] when filtered < 3 (Part 2.5)
  • NEVER drop link/image_url — LLM payload strips them, final_products keeps them (Part 4)
  • Robust compare engine with ordinal resolution + < 2 guard (Part 6)
  • Part 7 query safety: refine+category keyword → treated as new search
  • Strict output contract: summary/products/explanation/follow_up/state (Part 8)
  • Mandatory logging: query→tokens→result counts (Part 9)
  • Safe state serialisation
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Part 2 — Groq LLM Integration (query parsing only — safe, with fallback)
# ---------------------------------------------------------------------------

try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _Groq = None  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False

# Module-level Groq client initialised lazily from the environment variable.
# Used ONLY for query parsing — never for product generation.
_groq_parse_client = None


def _get_parse_client():
    """
    Return a cached Groq client for query parsing, or None if unavailable.
    Lazy init so import never blocks / crashes when the key is absent.
    """
    global _groq_parse_client
    if _groq_parse_client is not None:
        return _groq_parse_client
    if not _GROQ_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        _groq_parse_client = _Groq(api_key=api_key)
        logger.info("Groq parse client initialised.")
    except Exception as exc:
        logger.warning("Groq parse client init failed: %s", exc)
        _groq_parse_client = None
    return _groq_parse_client


def call_llm(prompt: str) -> str | None:
    """
    Safe, single-call LLM wrapper for query parsing.
    Returns raw text content or None on any failure.
    NEVER raises.
    """
    client = _get_parse_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("call_llm failed: %s", exc)
        return None


def parse_query_llm(query: str) -> dict | None:
    """
    Use Groq to extract structured shopping intent from the raw query.
    Returns a dict with keys: query, category, brand, keywords.
    Returns None on any failure — caller MUST fall back to rule-based parsing.

    LLM is used ONLY for normalisation / intent extraction.
    Products are NEVER generated here.
    """
    prompt = (
        "Extract structured shopping intent from this query.\n\n"
        "Return ONLY a valid JSON object with these exact keys:\n"
        '{"query": "...", "category": "...", "brand": "...", "keywords": []}\n\n'
        "Rules:\n"
        "- query: cleaned / corrected version of the user query\n"
        "- category: product category (e.g. 'phone', 'laptop', 'shoes') or null\n"
        "- brand: brand name if mentioned, else null\n"
        "- keywords: list of important product keywords\n"
        "No markdown, no explanation, ONLY the JSON object.\n\n"
        f'Query: "{query}"'
    )

    raw = call_llm(prompt)
    if not raw:
        return None

    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        # Validate essential key
        if not parsed.get("query"):
            parsed["query"] = query
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("parse_query_llm JSON decode failed: %s | raw=%r", exc, raw[:200])
        return None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# PART 1  —  UTILITY HELPERS
# ===========================================================================

def safe(val: Any) -> Any:
    """Return val as-is if meaningful, else 'Not specified'."""
    return val if val not in (None, "", "N/A") else "Not specified"


def safe_float(x: Any) -> float:
    """Convert any value to float safely; returns 0.0 on failure."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def clean_price(value: Any) -> Any:
    """
    Normalise a price value to a plain float for LLM consumption.
    Strips currency symbols, commas, and other non-numeric characters.
    Returns 0 on failure so the LLM always receives a numeric value.
    """
    try:
        return float(re.sub(r"[^\d.]", "", str(value)))
    except (ValueError, TypeError):
        return 0


def is_llm_available(client: Any) -> bool:
    """Returns True only when a client exists and GROQ_API_KEY is set."""
    return client is not None and bool(os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# 1.1  Query normalisation
# ---------------------------------------------------------------------------

_NORMALISE_REMOVE: frozenset = frozenset({
    "show", "me", "find", "give", "best", "please", "i",
    "want", "looking", "for", "a", "an", "the",
})


def normalize_query(q: str) -> str:
    """
    Lightweight query normalisation for the *effective* query fed to the engine.

    Steps:
      1. Lowercase and split.
      2. Remove filler words (_NORMALISE_REMOVE).
      3. Basic singular normalisation: tokens ending in 's' (len > 3) are stripped.
         e.g. 'bottles' → 'bottle', 'shoes' → 'shoe'
      4. Re-join and strip.
    """
    if not q or not q.strip():
        return ""

    tokens = q.lower().split()
    tokens = [t for t in tokens if t not in _NORMALISE_REMOVE]
    tokens = [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in tokens]
    return " ".join(tokens).strip()


# ---------------------------------------------------------------------------
# 1.2  Synonym expansion (Part 1.2)
# ---------------------------------------------------------------------------

SYNONYMS: Dict[str, List[str]] = {
    "bottle":     ["water bottle", "flask"],
    "flask":      ["bottle"],
    "phone":      ["mobile", "smartphone"],
    "mobile":     ["phone"],
    "shoes":      ["footwear", "sneakers"],
    "laptop":     ["notebook"],
    "headphones": ["headset", "earphones"],
    "headphone":  ["headset", "earphone"],
    "earphone":   ["headphones", "headset"],
    "sneakers":   ["shoes", "footwear"],
    "notebook":   ["laptop"],
    "smartphone": ["phone", "mobile"],
}


def expand_tokens(tokens: List[str]) -> List[str]:
    """
    Expand query tokens with synonyms (spec Part 1.2).
    Applied AFTER token extraction.
    """
    expanded: set = set(tokens)
    for t in tokens:
        if t in SYNONYMS:
            expanded.update(SYNONYMS[t])
    return list(expanded)


# ---------------------------------------------------------------------------
# 1.3  Soft token matching (Part 1.3 — replaces strict 'if t in product_text')
# ---------------------------------------------------------------------------

def soft_match(token: str, text: str) -> bool:
    """
    Soft substring match: returns True if token appears as a substring of any
    word in text, OR any word in text appears as a substring of token.
    Replaces strict equality checks for broader recall.
    """
    words = text.split()
    return any(token in w or w in token for w in words)


# ---------------------------------------------------------------------------
# 1.4  Garbage / vague query detection
# ---------------------------------------------------------------------------

def is_low_signal_query(q: str) -> bool:
    """
    Returns True when the query is too vague for meaningful retrieval.
    Triggers on:
      • Fewer than 1 token after stripping whitespace.
      • No alphabetic characters at all.
    """
    stripped = q.strip()
    if not stripped:
        return True
    tokens = stripped.split()
    if len(tokens) < 1:
        return True
    if not any(c.isalpha() for c in stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# Product normalisation
# ---------------------------------------------------------------------------

def normalize_products(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Coerce ProductResult objects (or any non-dict) into plain dicts.

    Field-name aliasing (additive — original keys are NEVER dropped):
      final_price  →  price      (engine uses final_price; UI expects price)
      image        →  image_url  (ProductResult.image; UI / templates expect image_url)

    Aliases are only written when the target key is absent, so dicts that
    already carry the canonical name are never overwritten.
    """
    normalised: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, dict):
            d = r.copy()
        else:
            try:
                d = r.__dict__.copy()
            except AttributeError:
                logger.warning("normalize_products: skipping unrecognised type %s", type(r))
                continue

        # final_price → price
        if "final_price" in d and "price" not in d:
            d["price"] = d["final_price"]

        # image → image_url
        if "image" in d and "image_url" not in d:
            d["image_url"] = d["image"]

        # image_url → image  (reverse alias for completeness)
        if "image_url" in d and "image" not in d:
            d["image"] = d["image_url"]

        normalised.append(d)
    return normalised


# ===========================================================================
# PART 2  —  DOMAIN-AGNOSTIC RELEVANCE ENGINE
# ===========================================================================

# --- Stopwords ---
_FILTER_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "for", "with", "under", "above", "below", "best",
    "cheap", "cheapest", "good", "top", "great", "new", "latest", "buy",
    "get", "find", "show", "me", "need", "want", "looking", "give",
    "in", "on", "at", "of", "to", "is", "are", "was", "be", "and",
    "or", "not", "very", "my", "i", "you", "it", "this", "that",
    "from", "by", "about", "some", "any", "please", "can", "could",
    "would", "within", "around", "price", "budget", "range", "rate",
    "rating", "rated", "popular", "brand", "upto", "up",
})

# --- Intent / product classification vocabularies ---
_PRIMARY_TERMS: frozenset = frozenset({
    "phone", "mobile", "smartphone", "shoes", "laptop", "tv",
    "bottle", "headphones", "watch", "spice",
})

_ACCESSORY_TERMS: frozenset = frozenset({
    "cover", "case", "charger", "cable", "adapter",
    "earphone", "protector", "battery", "card", "bag",
})

# --- Noise terms applied conditionally (only when absent from query) ---
_NOISE_TERMS: frozenset = frozenset({
    "accessory", "part", "refill", "attachment", "add-on",
    "case", "cover", "strap", "cable", "kit",
    # Extended for broader domain coverage
    "bag", "protector", "manual", "guide",
    "dictionary", "sleeve", "pouch", "holder", "insert", "liner",
    "lace", "laces", "cord", "charger", "adapter",
    "cleaning", "cloth", "stand", "mount", "hook", "hanger",
    "organiser", "organizer", "storage", "repair", "replacement",
    "spare", "parts",
})

# Minimum filtered count before the safety fallback kicks in
_MIN_RESULTS: int = 3


# ---------------------------------------------------------------------------
# 2.1  Token extraction
# ---------------------------------------------------------------------------

def extract_query_tokens(query: str) -> List[str]:
    """
    Tokenise, clean, and synonym-expand a query for relevance scoring.

    Steps:
      1. Lowercase.
      2. Remove punctuation; keep only alpha tokens.
      3. Remove stopwords.
      4. De-duplicate while preserving order.
      5. Synonym-expand (Part 1.2) — adds related terms for broader recall.
    """
    raw_tokens = re.findall(r"\b[a-z]+\b", query.lower())
    seen: set = set()
    tokens: List[str] = []
    for t in raw_tokens:
        if t not in _FILTER_STOPWORDS and t not in seen:
            seen.add(t)
            tokens.append(t)
    # Part 1.2 — expand with synonyms AFTER base extraction
    tokens = expand_tokens(tokens)
    return tokens


# ---------------------------------------------------------------------------
# 2.2  Product text normalisation
# ---------------------------------------------------------------------------

def build_product_text(product: Dict[str, Any]) -> str:
    """
    Build a single normalised lowercase text blob from searchable product fields.
    Combines: name · category · brand (+ sub_category when present).
    """
    parts = [
        str(product.get("name",         "") or ""),
        str(product.get("category",     "") or ""),
        str(product.get("brand",        "") or ""),
        str(product.get("sub_category", "") or ""),
    ]
    raw = " ".join(p for p in parts if p).lower()
    return re.sub(r"[^\w\s]", " ", raw)


# Backward-compatible alias
def get_product_text(product: Dict[str, Any]) -> str:  # noqa: D103
    return build_product_text(product)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def detect_query_type(tokens: List[str]) -> str:
    if any(t in _ACCESSORY_TERMS for t in tokens):
        return "accessory"
    return "primary"


def detect_product_type(product_text: str) -> str:
    if any(term in product_text for term in _ACCESSORY_TERMS):
        return "accessory"
    return "primary"


# ---------------------------------------------------------------------------
# 2.3  Lexical score
# ---------------------------------------------------------------------------

def compute_lexical_score(query_tokens: List[str], product_text: str) -> float:
    """
    Token-overlap ratio using soft_match (Part 1.3 + 2.1).
    Uses substring matching instead of strict equality for broader recall.
    """
    if not query_tokens:
        return 0.0
    overlap = sum(1 for t in query_tokens if soft_match(t, product_text))
    return overlap / len(query_tokens)


# ---------------------------------------------------------------------------
# 2.4  Soft category signal
# ---------------------------------------------------------------------------

def compute_category_score(query_tokens: List[str], product_text: str) -> float:
    """
    Soft category alignment bonus (Part 2.2): 0.2 when any token soft-matches, else 0.0.
    Uses soft_match instead of strict substring check.
    """
    if any(soft_match(t, product_text) for t in query_tokens):
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# 2.5  Intent-aware penalty (GENERIC)
# ---------------------------------------------------------------------------

def apply_intent_penalty(score: float, tokens: List[str], text: str) -> float:
    """
    Penalise accessory products when the user asked for a primary item.

    User wants accessory  → no penalty.
    Product is accessory but user did NOT ask for one → score *= 0.6  (Part 2.3)

    Uses soft_match for noise detection (Part 1.3).
    """
    user_wants_accessory  = any(soft_match(t, " ".join(_NOISE_TERMS)) for t in tokens)
    product_is_accessory  = any(soft_match(t, text) for t in _NOISE_TERMS)

    if not user_wants_accessory and product_is_accessory:
        score *= 0.6   # relaxed from 0.4 per spec Part 2.3
    return score


# ---------------------------------------------------------------------------
# 2.6  Final scoring
# ---------------------------------------------------------------------------

def compute_relevance_score(
    query_tokens: List[str],
    product_text: str,
    product: Dict[str, Any],
) -> float:
    """
    Full hybrid relevance score combining upstream engine score with lexical
    and category signals, then applying intent penalty.

    Formula (per spec Part 2.6):
        score = 0.6 * engine_score + 0.3 * lexical_score + category_score
        → apply_intent_penalty

    Clamped to [0.0, 1.0].
    """
    if not query_tokens:
        return 1.0  # no meaningful tokens → treat as fully relevant

    engine_score: float = safe_float(
        product.get("final_score") or product.get("similarity_score") or 0
    )
    engine_score = max(0.0, min(1.0, engine_score))

    lexical_score  = compute_lexical_score(query_tokens, product_text)
    category_score = compute_category_score(query_tokens, product_text)

    score = (0.6 * engine_score) + (0.3 * lexical_score) + category_score
    score = apply_intent_penalty(score, query_tokens, product_text)

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 2.7  Adaptive threshold (BALANCED — per spec)
# ---------------------------------------------------------------------------

def compute_dynamic_threshold(query_tokens: List[str], _candidates: List[Any] = None) -> float:
    """
    Balanced adaptive threshold (spec Part 2.7):
      ≤ 1 token  → 0.25
      ≤ 3 tokens → 0.35
      > 3 tokens → 0.45
    """
    n = len(query_tokens)
    if n <= 1:
        return 0.25
    elif n <= 3:
        return 0.35
    return 0.45


# ---------------------------------------------------------------------------
# 2.8 + 2.9  Filtering with safe fallback AND retry-without-penalties
# ---------------------------------------------------------------------------

def _score_products(
    query_tokens: List[str],
    original: List[Dict[str, Any]],
    apply_penalties: bool = True,
) -> List[Tuple[float, float, str, Dict[str, Any]]]:
    """
    Pre-compute (relevance_score, engine_score, name, product) tuples.
    When apply_penalties=False, uses pure lexical+engine scoring (no noise penalty).
    """
    enriched: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for product in original:
        text = build_product_text(product)

        if apply_penalties:
            rel_score = compute_relevance_score(query_tokens, text, product)
        else:
            # Retry path: no intent / noise penalty — pure hybrid score
            engine_score = max(0.0, min(1.0, safe_float(
                product.get("final_score") or product.get("similarity_score") or 0
            )))
            lexical_score  = compute_lexical_score(query_tokens, text)
            category_score = compute_category_score(query_tokens, text)
            rel_score = max(0.0, min(1.0,
                (0.6 * engine_score) + (0.3 * lexical_score) + category_score
            ))

        engine_score_raw = safe_float(
            product.get("final_score") or product.get("similarity_score") or 0
        )
        name = str(product.get("name", "")).lower()
        enriched.append((rel_score, engine_score_raw, name, product))

    enriched.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return enriched


def detect_category(tokens: List[str]) -> Optional[str]:
    """
    Detect dominant category intent from query tokens (Part 3).
    Returns the first recognised category token, or None.
    """
    _CATEGORY_TOKENS = {
        "bottle", "phone", "laptop", "shoes", "headphone", "headphones",
        "mobile", "tv", "watch", "camera", "bag", "flask",
    }
    for t in tokens:
        if t in _CATEGORY_TOKENS:
            return t
    return None


def apply_relevance_filter(
    results: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """
    Post-retrieval relevance filter with multi-stage fallback (Parts 2.4, 2.5, 3, 5, 9).

    PASS 1 — Full scoring with penalties + category boost.
    PASS 2 — Retry WITHOUT penalties when pass 1 yields 0.
    PASS 3 — Engine-only fallback sorted by final_score when pass 2 still yields 0.
    NEVER EMPTY RULE — if filtered < _MIN_RESULTS, return original[:5].

    Spec Part 9 logging on every stage.
    """
    if not results:
        return results

    original = [p for p in results if isinstance(p, dict) and p.get("name")]
    if not original:
        return list(results)

    query_tokens = extract_query_tokens(query)

    # Part 9 — mandatory logging
    logger.info("Query: %s → %s", query, " ".join(query_tokens))
    logger.info("Tokens: %s", query_tokens)

    if not query_tokens:
        logger.debug("apply_relevance_filter: no tokens in '%s' — skipping filter", query)
        logger.info("Results: %d → %d", len(original), len(original))
        return original

    threshold = compute_dynamic_threshold(query_tokens)

    # Part 3 — detect category for boost
    category_token = detect_category(query_tokens)

    def _apply_category_boost(enriched_list):
        """Apply +0.1 boost when product matches detected category token."""
        if not category_token:
            return enriched_list
        boosted = []
        for rel_score, engine_score, name, product in enriched_list:
            ptext = build_product_text(product)
            if soft_match(category_token, ptext):
                rel_score = min(1.0, rel_score + 0.1)
            boosted.append((rel_score, engine_score, name, product))
        boosted.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return boosted

    # --- Pass 1: full scoring with penalties + category boost ---
    enriched = _score_products(query_tokens, original, apply_penalties=True)
    enriched = _apply_category_boost(enriched)
    filtered = [product for rel_score, _, _, product in enriched if rel_score >= threshold]

    logger.info(
        "Relevance filtering (pass 1): %d → %d | threshold=%.2f | query='%s'",
        len(original), len(filtered), threshold, query,
    )

    # --- Pass 2: retry WITHOUT penalties if pass 1 empty (Part 2.4) ---
    if len(filtered) == 0:
        logger.warning(
            "No results after pass-1 filtering — retrying without penalties. "
            "threshold=%.2f | query='%s'", threshold, query,
        )
        enriched_retry = _score_products(query_tokens, original, apply_penalties=False)
        enriched_retry = _apply_category_boost(enriched_retry)
        filtered = [product for rel_score, _, _, product in enriched_retry if rel_score >= threshold]

        logger.info(
            "Relevance filtering (pass 2, no penalties): %d → %d | threshold=%.2f",
            len(original), len(filtered), threshold,
        )

    # --- Pass 3: engine-only fallback (Part 2.4) ---
    if len(filtered) == 0:
        logger.warning(
            "No results after pass-2 filtering — falling back to engine score ranking. "
            "query='%s'", query,
        )
        filtered = sorted(
            original, key=lambda x: x.get("final_score", 0), reverse=True
        )[:5]
        logger.info(
            "Results (pass 3 engine fallback): %d → %d", len(original), len(filtered)
        )

    # --- Part 2.5 / NEVER EMPTY RULE ---
    if len(filtered) < _MIN_RESULTS:
        logger.warning(
            "Filtered count (%d) below minimum (%d). Applying safe fallback.",
            len(filtered), _MIN_RESULTS,
        )
        filtered = original[:5]

    # Part 9 — final results count
    logger.info("Results: %d → %d", len(original), len(filtered))
    return filtered


# Backward-compatible alias
def filter_relevance(
    results: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Alias for apply_relevance_filter — preserves backward compatibility."""
    return apply_relevance_filter(results, query)


# ===========================================================================
# PART 3  —  METADATA PROPAGATION  (LINK-SAFE)
# ===========================================================================

# Keys excluded from the LLM payload only — NEVER from final_products
_LLM_STRIP_KEYS: frozenset = frozenset({
    "link", "image_url", "image", "product_text", "embedding_text", "description",
})


def build_llm_payload(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a noise-free payload for the LLM (top 3 products).
    Strips URL/image/text-blob fields.  price is normalised to float.
    NEVER used for final_products — those keep all metadata.
    """
    return [
        {
            **{k: v for k, v in p.items() if k not in _LLM_STRIP_KEYS},
            "price": clean_price(p.get("price", p.get("final_price"))),
        }
        for p in results[:3]
    ]


# ===========================================================================
# PART 4  —  COMPARISON ENGINE  (ROBUST)
# ===========================================================================

def compare_products(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares two products from results.
    Uses ordinal resolution; falls back to first two items.
    ZERO hallucination — only data from results.
    """
    if not results:
        return {
            "summary":   "No products available to compare. Please search first.",
            "reasoning": [],
        }

    # Part 6 — comparison stability: need at least 2 results
    if len(results) < 2:
        return {
            "summary":   "Need at least two products to compare. Try broadening your search.",
            "reasoning": [],
        }

    q = query.lower()

    # Ordinal resolution (spec Part 4)
    idx1, idx2 = 0, 1

    if "second" in q or " 2 " in q or q.endswith(" 2"):
        idx1 = 1
    if "third" in q or " 3 " in q or q.endswith(" 3"):
        idx2 = 2

    p1_idx = min(idx1, len(results) - 1)
    p2_idx = min(idx2, len(results) - 1)

    # Guard: ensure different indices
    if p1_idx == p2_idx:
        p2_idx = 1 if p1_idx == 0 else 0

    p1 = results[p1_idx]
    p2 = results[p2_idx]

    def _get(p: dict, key: str) -> str:
        return str(safe(p.get(key)))

    lines: List[str] = []

    a_name  = _get(p1, "name")
    b_name  = _get(p2, "name")

    # Price
    a_price = _get(p1, "price")
    b_price = _get(p2, "price")
    lines.append(f"Price — {a_name}: ₹{a_price}  |  {b_name}: ₹{b_price}")

    try:
        price_a = float(re.sub(r"[^\d.]", "", a_price))
        price_b = float(re.sub(r"[^\d.]", "", b_price))
        cheaper = a_name if price_a <= price_b else b_name
        lines.append(f"Cheaper option: {cheaper}")
    except (ValueError, TypeError):
        pass

    # Ratings
    lines.append(
        f"Ratings — {a_name}: {_get(p1, 'ratings')}  |  {b_name}: {_get(p2, 'ratings')}"
    )

    # Brand
    lines.append(
        f"Brand — {a_name}: {_get(p1, 'brand')}  |  {b_name}: {_get(p2, 'brand')}"
    )

    # Factual recommendation
    recommendation = "Not enough data for a factual recommendation."
    try:
        ra = float(str(safe(p1.get("ratings"))))
        rb = float(str(safe(p2.get("ratings"))))
        if ra > rb:
            recommendation = f"{a_name} has a higher rating ({ra} vs {rb})."
        elif rb > ra:
            recommendation = f"{b_name} has a higher rating ({rb} vs {ra})."
        else:
            recommendation = "Both products have equal ratings; consider price and brand preference."
    except (ValueError, TypeError):
        pass

    return {
        "summary":   f"Comparison: {a_name} vs {b_name}",
        "reasoning": lines + [recommendation],
    }


# ===========================================================================
# STATE MODEL
# ===========================================================================

class ConversationState(BaseModel):
    last_query:   Optional[str]         = None
    last_results: List[Dict[str, Any]]  = Field(default_factory=list)
    filters:      Dict[str, Any]        = Field(default_factory=dict)
    history:      List[str]             = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(**data)
        except Exception:
            return cls()

    def push_history(self, entry: str) -> None:
        self.history.append(entry)
        self.history = self.history[-5:]


# ===========================================================================
# INTENT CLASSIFIER  (rule-first, no LLM)
# ===========================================================================

_COMPARE_KEYWORDS = {"compare", "vs", "versus", "difference", "differences", "diff", "better"}
_EXPLAIN_KEYWORDS = {"why", "reason", "explain", "because", "how come", "rationale"}
_REFINE_SIGNALS   = {
    "under", "above", "below", "cheaper", "expensive", "more", "less",
    "better", "worst", "best", "higher", "lower", "within", "budget",
    "price", "around", "upto", "up to",
}

_CATEGORY_HINTS: Dict[str, str] = {
    "shoes":     "Are you looking for running, casual, or formal shoes?",
    "phone":     "Are you looking for a budget, mid-range, or flagship smartphone?",
    "laptop":    "Are you looking for a laptop for gaming, work, or general use?",
    "bag":       "Are you looking for a handbag, backpack, or travel bag?",
    "watch":     "Are you looking for a smartwatch or a traditional timepiece?",
    "headphone": "Are you looking for wired, wireless, or noise-cancelling headphones?",
    "camera":    "Are you looking for a DSLR, mirrorless, or point-and-shoot camera?",
    "tv":        "What screen size or budget range are you considering for the TV?",
    "shirt":     "Are you looking for a formal, casual, or sports shirt?",
}


def classify_intent(query: str, state: ConversationState) -> str:
    """
    Deterministic rule-based intent classification.
    Priority: compare > explain > refine > clarify > search
    """
    if not query or not query.strip():
        return "clarify"

    tokens = set(re.findall(r"\b\w+\b", query.lower()))

    if tokens & _COMPARE_KEYWORDS:
        return "compare"

    if tokens & _EXPLAIN_KEYWORDS:
        return "explain"

    if state.last_query and (tokens & _REFINE_SIGNALS):
        if any(t in tokens for t in _PRIMARY_TERMS):
            return "search"
        return "refine"

    words = query.strip().split()
    if len(words) < 2 and not any(k in query.lower() for k in _CATEGORY_HINTS):
        return "clarify"

    meaningful = [w for w in words if re.search(r"[a-zA-Z]{2,}", w)]
    if not meaningful:
        return "clarify"

    return "search"


# ===========================================================================
# CLARIFICATION ENGINE
# ===========================================================================

def needs_clarification(query: str) -> Optional[str]:
    """Returns a follow-up question when the query is too vague, or None."""
    if not query or not query.strip():
        return "What product are you looking for today?"

    words = query.strip().split()

    if len(words) < 2:
        lower = query.lower().strip()
        for keyword, question in _CATEGORY_HINTS.items():
            if keyword in lower:
                return question
        return f"Could you tell me more about what you're looking for with '{query}'?"

    meaningful = [w for w in words if re.search(r"[a-zA-Z]{2,}", w)]
    if not meaningful:
        return "I didn't quite catch that. Could you describe the product you have in mind?"

    return None


# ===========================================================================
# QUERY REFINEMENT
# ===========================================================================

def refine_query(prev_query: str, new_query: str) -> str:
    """
    Merges a refinement constraint into the previous query.
    Never overwrites original intent.
    """
    if not prev_query or not prev_query.strip():
        return new_query.strip()

    prev = prev_query.strip()
    new  = new_query.strip()

    if new.lower() in prev.lower():
        return prev

    return f"{prev} {new}"


# ===========================================================================
# EXPLANATION GENERATOR  (single LLM call, strict JSON)
# ===========================================================================

_EXPLANATION_SYSTEM_PROMPT = """\
You are a factual product explanation assistant.
You ONLY use the product data provided to you — never invent specs, features, or products.
Respond ONLY with a valid JSON object matching this schema exactly:
{
  "summary": "<one-sentence overview>",
  "reasoning": ["<point 1>", "<point 2>", "<point 3 optional>"]
}
No markdown, no prose outside the JSON object.
"""


def generate_explanation(
    intent: str,
    query: str,
    results: List[Dict[str, Any]],
    client: Any,
) -> Dict[str, Any]:
    """
    Calls the Groq LLM once in strict JSON mode to produce a grounded explanation.
    Falls back gracefully on any failure. Single LLM call — no retries.
    """
    _fallback = {
        "summary":   f"Here are the top results for '{query}'.",
        "reasoning": [],
    }

    if not results:
        return {
            "summary":   "We couldn't find products matching your query. Please try different keywords.",
            "reasoning": [],
        }

    if not is_llm_available(client):
        names = ", ".join(str(safe(p.get("name"))) for p in results[:3])
        return {
            "summary":   f"Here are the top results for '{query}': {names}.",
            "reasoning": [],
        }

    # Build noise-free LLM payload (links/images stripped — Part 3)
    llm_payload = build_llm_payload(results)

    user_content = (
        f"User query: {query}\n"
        f"Intent: {intent}\n"
        f"Products:\n{json.dumps(llm_payload, ensure_ascii=False, indent=2)}\n\n"
        "Based ONLY on the products above, generate a summary and reasoning."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _EXPLANATION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )

        raw    = response.choices[0].message.content
        parsed = json.loads(raw)

        summary   = str(parsed.get("summary", "")).strip() or "No summary available."
        reasoning = parsed.get("reasoning", [])
        if not isinstance(reasoning, list):
            reasoning = []
        reasoning = [str(r) for r in reasoning]

        return {"summary": summary, "reasoning": reasoning}

    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        return _fallback


# ===========================================================================
# RESPONSE BUILDER
# ===========================================================================

def build_response(
    intent: str,
    results: List[Dict[str, Any]],
    explanation: Dict[str, Any],
    follow_up: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assembles the final response dict.
    `results` must be raw (post-filter) engine output — untouched metadata.
    """
    raw_summary   = explanation.get("summary",   "") if explanation else ""
    raw_reasoning = explanation.get("reasoning", []) if explanation else []
    return {
        "summary":     raw_summary or "Here are some relevant products.",
        "products":    results,
        "explanation": raw_reasoning if raw_reasoning else None,
        "follow_up":   follow_up if follow_up else None,
    }


# ===========================================================================
# SAFETY GUARD
# ===========================================================================

def _safety_check(
    response: Dict[str, Any],
    allowed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ensures every product in the response exists in allowed_results.
    Removes any product that wasn't returned by the engine.
    """
    if not allowed_results:
        response["products"] = []
        return response

    allowed_ids:   set = {p.get("id")   for p in allowed_results if p.get("id")}
    allowed_names: set = {str(p.get("name", "")).lower() for p in allowed_results if p.get("name")}

    safe_products = []
    for p in response.get("products", []):
        pid  = p.get("id")
        name = str(p.get("name", "")).lower()
        if pid in allowed_ids or name in allowed_names:
            safe_products.append(p)

    response["products"] = safe_products
    return response


# ===========================================================================
# HARD CONSTRAINT FILTER
# ===========================================================================

def apply_hard_constraints(
    results: List[Dict[str, Any]],
    parsed_query: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Final safety filter: enforces brand and price constraints that the upstream
    pipeline (FAISS / ranking fallback) may have violated.
    """
    if not parsed_query:
        return results

    brand     = parsed_query.get("brand")
    max_price = parsed_query.get("max_price")

    if not brand and not max_price:
        return results

    filtered: List[Dict[str, Any]] = []
    for p in results:
        name = str(p.get("name", "")).lower()

        if brand and brand.lower() not in name:
            continue

        if max_price is not None:
            try:
                price_val = float(
                    str(p.get("price", "")).replace("₹", "").replace(",", "")
                )
                if price_val > max_price:
                    continue
            except (ValueError, TypeError):
                continue

        filtered.append(p)

    return filtered


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _terminal_response(
    summary: str,
    follow_up: Optional[str],
    state: ConversationState,
) -> Dict[str, Any]:
    """Returns a response with no products (clarification / error path)."""
    # Part 5: safe state serialisation
    safe_state = state.to_dict() if hasattr(state, "to_dict") else (state or {})
    return {
        "summary":     summary,
        "products":    [],
        "explanation": [],
        "follow_up":   follow_up,
        "state":       safe_state,
    }


# ===========================================================================
# PART 5  —  MAIN ENTRY FUNCTION
# ===========================================================================

def handle_user_query(
    query: str,
    state: dict,
    client: Any,
) -> Dict[str, Any]:
    """
    Orchestrates one full turn of the conversation layer.

    Args:
        query:  Raw user input string.
        state:  Serialised ConversationState dict (may be empty {}).
        client: Groq (or compatible) API client instance.

    Returns:
        A dict with keys: summary, products, explanation, follow_up, state.
    """

    # ------------------------------------------------------------------ #
    # Step 1 — Rehydrate state
    # ------------------------------------------------------------------ #
    conv_state = ConversationState.from_dict(state)

    # ------------------------------------------------------------------ #
    # Step 2 — Validate / sanitise input
    # ------------------------------------------------------------------ #
    if not isinstance(query, str) or not query.strip():
        return {
            "summary":     "Please enter a valid query.",
            "products":    [],
            "explanation": None,
            "follow_up":   "Try something like 'phones under 20000'.",
            "state":       conv_state.to_dict(),
        }

    query = query.strip()

    # ------------------------------------------------------------------ #
    # Part 1.3 — Garbage / low-signal guard (before intent classification)
    # ------------------------------------------------------------------ #
    if is_low_signal_query(query):
        return _terminal_response(
            summary="I need a bit more detail to help you.",
            follow_up="Could you describe the product you have in mind?",
            state=conv_state,
        )

    # ------------------------------------------------------------------ #
    # Step 3 — Classify intent
    # ------------------------------------------------------------------ #
    intent = classify_intent(query, conv_state)

    # ------------------------------------------------------------------ #
    # Step 4 — Handle clarify immediately (no retrieval needed)
    # ------------------------------------------------------------------ #
    if intent == "clarify":
        follow_up = needs_clarification(query)
        return _terminal_response(
            summary="I need a bit more detail to help you.",
            follow_up=follow_up or "Could you provide more details?",
            state=conv_state,
        )

    # ------------------------------------------------------------------ #
    # Part 1.2 — Determine effective query (query corruption prevention)
    # ------------------------------------------------------------------ #
    if not conv_state.last_query:
        effective_query = query
    elif intent == "refine":
        effective_query = refine_query(conv_state.last_query, query)
    else:
        effective_query = query

    # Part 1.1 — Normalise AFTER building effective_query
    effective_query = normalize_query(effective_query)

    # Part 7 — Query safety: if intent is "refine" but new query contains a
    # category keyword, treat as a fresh search to prevent query corruption.
    if intent == "refine":
        _new_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if _new_tokens & set(_PRIMARY_TERMS):
            logger.info(
                "Part 7: refine query contains category keyword — escalating to search. "
                "query=%r", query,
            )
            intent = "search"
            effective_query = normalize_query(query)

    logger.info("Query: %s | Effective: %s", query, effective_query)

    # ------------------------------------------------------------------ #
    # Step 6 — Retrieve products (compare / explain reuse state if present)
    # ------------------------------------------------------------------ #
    results: List[Dict[str, Any]] = []

    if intent in ("compare", "explain") and conv_state.last_results:
        # Reuse existing state results — already normalised
        results = normalize_products(conv_state.last_results)
    else:
        try:
            from src.retrieval.engine import search_with_filters  # noqa: PLC0415
            raw     = search_with_filters(effective_query, client, top_n=5) or []
            results = normalize_products(raw)
        except Exception as exc:
            logger.error("Engine retrieval failed: %s", exc)
            results = []

        # Post-retrieval relevance filter (Parts 2.8 + 2.9)
        results = filter_relevance(results, effective_query)

        # Hard cap — 5 products max
        results = results[:5]

        logger.info("Results: %d → %d", len(normalize_products(raw)) if 'raw' in dir() else 0, len(results))

        # Hard constraint enforcement
        pre_constraint = results
        parsed_query: Optional[Dict[str, Any]] = None

        # Part 2 — LLM query parsing (primary) with rule-based fallback (MANDATORY)
        try:
            parsed_query = parse_query_llm(query)
            if parsed_query:
                logger.info("LLM parse succeeded: %s", parsed_query)
        except Exception as _llm_exc:
            logger.warning("parse_query_llm raised unexpectedly: %s", _llm_exc)
            parsed_query = None

        if not parsed_query:
            # Fallback: use rule-based parser from query_parser module
            try:
                from src.llm.query_parser import parse_query  # noqa: PLC0415
                parsed_query = parse_query(query, client)
            except Exception:
                pass

        results = apply_hard_constraints(results, parsed_query)

        logger.info(
            "Hard constraint filtering: %d → %d",
            len(pre_constraint), len(results),
        )

        if not results:
            logger.warning("Hard constraints removed all results. Returning relaxed fallback.")
            results = pre_constraint[:3]

        # Update state
        conv_state.last_query   = effective_query
        conv_state.last_results = results
        conv_state.push_history(effective_query)

    # ------------------------------------------------------------------ #
    # Step 7 — Empty result protection
    # ------------------------------------------------------------------ #
    if not results:
        try:
            from src.retrieval.engine import search_with_filters  # noqa: PLC0415
            raw     = search_with_filters(query, client, top_n=5) or []
            results = normalize_products(raw)[:5]
        except Exception as exc:
            logger.error("Fallback retrieval failed: %s", exc)
            results = []

        if not results:
            return {
                "summary":     "I couldn't find results. Try a different query.",
                "products":    [],
                "explanation": [],
                "follow_up":   "Try specifying a category or price range.",
                "state":       conv_state.to_dict(),
            }

    # ------------------------------------------------------------------ #
    # Step 8 — Generate output
    # ------------------------------------------------------------------ #
    if intent == "compare":
        explanation = compare_products(query, results)
        follow_up   = None
    else:
        explanation = generate_explanation(intent, effective_query, results, client)
        follow_up   = None

    # ------------------------------------------------------------------ #
    # Part 3 — final_products retains ALL metadata  (link / image_url safe)
    # ------------------------------------------------------------------ #
    final_products = results  # NEVER re-create; NEVER filter fields

    # ------------------------------------------------------------------ #
    # Step 9 — Build response
    # ------------------------------------------------------------------ #
    response = build_response(intent, final_products, explanation, follow_up=follow_up)

    # ------------------------------------------------------------------ #
    # Step 10 — Safety guard (no hallucination)
    # ------------------------------------------------------------------ #
    response = _safety_check(response, results)

    if response["products"] and results:
        result_names = {str(r.get("name", "")).lower() for r in results}
        for p in response["products"]:
            p_name = str(p.get("name", "")).lower()
            if p_name and p_name not in result_names:
                logger.error(
                    "Safety violation: product '%s' not in engine results — removing.", p_name
                )
        response["products"] = [
            p for p in response["products"]
            if str(p.get("name", "")).lower() in result_names
        ]

    # ------------------------------------------------------------------ #
    # Part 5 — Attach serialised state and return
    # ------------------------------------------------------------------ #
    safe_state = conv_state.to_dict() if hasattr(conv_state, "to_dict") else (conv_state or {})
    response["state"] = safe_state
    return response


# ===========================================================================
# Module self-test  (run with:  python responder.py)
# ===========================================================================

if __name__ == "__main__":
    import types
    import sys

    print("=" * 60)
    print("UNIT TESTS — relevance filter functions")
    print("=" * 60)

    # --- normalize_query ---
    assert normalize_query("show me best bottles") == "bottle", \
        f"FAIL normalize_query: {normalize_query('show me best bottles')}"
    assert normalize_query("") == "", "FAIL: empty query"
    print(f"✔ normalize_query: 'show me best bottles' → '{normalize_query('show me best bottles')}'")

    # --- is_low_signal_query ---
    assert is_low_signal_query("") is True, "FAIL: empty should be low-signal"
    assert is_low_signal_query("asdfgh12!!") is False, "FAIL: has alpha chars"
    assert is_low_signal_query("12345") is True, "FAIL: no alpha chars"
    print("✔ is_low_signal_query")

    # --- extract_query_tokens ---
    t1 = extract_query_tokens("best running shoes under 2000")
    assert t1 == ["running", "shoes"], f"FAIL token extraction: {t1}"
    print(f"✔ extract_query_tokens: {t1}")

    # --- compute_dynamic_threshold (balanced) ---
    assert compute_dynamic_threshold([]) == 0.25,   "FAIL: 0 tokens → 0.25"
    assert compute_dynamic_threshold(["shoes"]) == 0.25, "FAIL: 1 token → 0.25"
    assert compute_dynamic_threshold(["running", "shoes"]) == 0.35, "FAIL: 2 tokens → 0.35"
    assert compute_dynamic_threshold(["a", "b", "c", "d"]) == 0.45, "FAIL: 4 tokens → 0.45"
    print("✔ compute_dynamic_threshold (balanced thresholds)")

    # --- apply_intent_penalty ---
    score = apply_intent_penalty(1.0, ["shoes"], "shoe bag case")
    assert score < 1.0, f"FAIL: accessory should be penalised: {score}"
    score_ok = apply_intent_penalty(1.0, ["shoes"], "nike running shoes")
    assert score_ok == 1.0, f"FAIL: primary product should not be penalised"
    print("✔ apply_intent_penalty")

    # --- filter_relevance ---
    noisy_results = [
        {"id": "1", "name": "Nike Air Max 90",   "category": "Running Shoes", "brand": "Nike",   "ratings": 4.6},
        {"id": "2", "name": "Adidas Ultraboost", "category": "Running Shoes", "brand": "Adidas", "ratings": 4.5},
        {"id": "3", "name": "Shoe Bag Deluxe",   "category": "Bag",           "brand": "Generic","ratings": 3.0},
        {"id": "4", "name": "Lace Kit Pro",      "category": "Accessories",   "brand": "Generic","ratings": 2.8},
        {"id": "5", "name": "Sneaker Dictionary", "category": "Books",        "brand": "Oxford", "ratings": 4.2},
    ]
    filtered = filter_relevance(noisy_results, "running shoes")
    filtered_names = [p["name"] for p in filtered]
    assert "Nike Air Max 90"   in filtered_names, "FAIL: Nike should pass"
    assert "Adidas Ultraboost" in filtered_names, "FAIL: Adidas should pass"
    print(f"✔ filter_relevance (noisy): {len(noisy_results)} → {len(filtered)}: {filtered_names}")

    # --- safety fallback: never over-filter ---
    unrelated = [{"id": "1", "name": "Widget A", "category": "Misc", "brand": "X", "ratings": 2.0}]
    fb = filter_relevance(unrelated, "shoes")
    assert fb == unrelated, "FAIL: safety fallback should return originals"
    print("✔ filter_relevance (safety fallback): correctly returned originals")

    # --- empty query → no filtering ---
    as_is = filter_relevance(noisy_results, "the for with")
    assert as_is == noisy_results, "FAIL: all-stopword query should skip filtering"
    print("✔ filter_relevance (empty tokens): skipped correctly")

    # --- compare_products ---
    products_2 = [
        {"name": "HyperX Cloud II", "price": 4999, "ratings": 4.5, "brand": "HyperX"},
        {"name": "Sony WH-1000XM5", "price": 24999, "ratings": 4.7, "brand": "Sony"},
    ]
    cmp = compare_products("compare first vs second", products_2)
    assert cmp["summary"].startswith("Comparison:"), f"FAIL compare summary: {cmp['summary']}"
    assert len(cmp["reasoning"]) > 0, "FAIL: compare should produce reasoning"
    print(f"✔ compare_products: {cmp['summary']}")

    cmp_single = compare_products("compare", [{"name": "Only One", "price": 999}])
    assert "Need at least two" in cmp_single["summary"], "FAIL: single product compare"
    print("✔ compare_products (single item guard)")

    print()
    print("=" * 60)
    print("INTEGRATION TESTS — handle_user_query")
    print("=" * 60)

    # ---- Minimal mock client ----
    class _MockCompletion:
        class _Choice:
            class _Msg:
                content = json.dumps({
                    "summary": "These are highly-rated gaming headphones.",
                    "reasoning": ["Product A has a 4.5 rating.", "Product B is cheaper."]
                })
            message = _Msg()
        choices = [_Choice()]

    class _MockChat:
        class completions:
            @staticmethod
            def create(**_kwargs):
                return _MockCompletion()

    mock_client = types.SimpleNamespace(chat=_MockChat)

    # ---- ProductResult stub ----
    class ProductResult:
        def __init__(self, id, name, category, final_price, ratings, brand):
            self.id          = id
            self.name        = name
            self.category    = category
            self.final_price = final_price
            self.ratings     = ratings
            self.brand       = brand

    # ---- Mock engine: 5 results, 2 of which are noise ----
    engine_mock = types.ModuleType("src.retrieval.engine")
    engine_mock.search_with_filters = lambda q, c, top_n=5: [
        ProductResult("1", "HyperX Cloud II",      "Gaming Headphones", 4999,  4.5, "HyperX"),
        ProductResult("2", "Sony WH-1000XM5",      "Headphones",        24999, 4.7, "Sony"),
        ProductResult("3", "Boat Rockerz 550",     "Headphones",        1799,  4.1, "Boat"),
        ProductResult("4", "Headphone Carry Case", "Accessories",       299,   3.8, "Generic"),
        ProductResult("5", "Ear Cushion Kit",      "Accessories",       149,   3.5, "Generic"),
    ]

    src_mock       = types.ModuleType("src")
    retrieval_mock = types.ModuleType("src.retrieval")
    sys.modules["src"]                  = src_mock
    sys.modules["src.retrieval"]        = retrieval_mock
    sys.modules["src.retrieval.engine"] = engine_mock

    print("\n=== TURN 1: search with noisy FAISS results ===")
    r1 = handle_user_query("gaming headphones", {}, mock_client)
    names_1 = [p["name"] for p in r1["products"]]
    print(f"Products returned: {names_1}")
    assert r1["products"], "FAIL: products should not be empty"
    assert "price" in r1["products"][0], "FAIL: price field missing"
    assert "Headphone Carry Case" not in names_1, "FAIL: carry case should be filtered"
    assert "Ear Cushion Kit"      not in names_1, "FAIL: ear cushion kit should be filtered"

    print("\n=== TURN 2: refine ===")
    r2 = handle_user_query("under 5000", r1["state"], mock_client)
    print(f"Products: {[p['name'] for p in r2['products']]}")

    print("\n=== TURN 3: compare first vs second ===")
    r3 = handle_user_query("compare first vs second", r2["state"], mock_client)
    assert r3["summary"].startswith("Comparison:"), f"FAIL: {r3['summary']}"
    print(f"Summary: {r3['summary']}")

    print("\n=== TURN 4: category keyword routes to search ===")
    r4 = handle_user_query("shoes", {}, mock_client)
    assert r4["products"] is not None
    print(f"Products: {[p['name'] for p in r4['products']]}")

    print("\n=== TURN 5: empty input ===")
    r5 = handle_user_query("", {}, mock_client)
    assert r5["products"] == [], "FAIL: empty input should yield no products"

    print("\n=== TURN 6: nonsense ===")
    r6 = handle_user_query("asdfgh12!!", {}, mock_client)
    print(f"Follow-up: {r6['follow_up']}")

    print("\n=== TURN 7: compare with only 1 result in state ===")
    thin_state = {
        "last_query": "headphones",
        "last_results": [
            {"id": "1", "name": "HyperX Cloud II", "category": "Headphones",
             "price": 4999, "ratings": 4.5, "brand": "HyperX"}
        ],
        "filters": {}, "history": []
    }
    r7 = handle_user_query("compare first vs second", thin_state, mock_client)
    assert r7["summary"], "FAIL: should always return a non-empty summary"
    print(f"Summary: {r7['summary']}")

    print("\n=== TURN 8: LLM path with GROQ_API_KEY set ===")
    os.environ["GROQ_API_KEY"] = "test-key"
    r8 = handle_user_query("gaming headphones", {}, mock_client)
    assert r8["summary"], "FAIL: LLM path should produce a summary"
    print(f"Summary: {r8['summary']}")

    print("\n=== TURN 9: low-signal numeric query ===")
    r9 = handle_user_query("12345", {}, mock_client)
    assert r9["products"] == [], "FAIL: numeric-only query should yield no products"
    print(f"Follow-up: {r9['follow_up']}")

    print("\n=== TURN 10: normalize_query in effective_query ===")
    r10 = handle_user_query("show me best bottles", {}, mock_client)
    assert r10["state"]["last_query"] == "bottle", \
        f"FAIL: effective_query should be normalised, got: {r10['state']['last_query']}"
    print(f"Effective query stored: {r10['state']['last_query']}")

    print("\n✅ All assertions passed.")