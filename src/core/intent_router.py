"""
intent_router.py

Deterministic, zero-LLM intent classifier.
Maps (validated slots, current state, raw query) → a single intent label.

Architecture: Validator + State Manager → [THIS MODULE] → Engine → Filters → Ranking

Returns EXACTLY one of: "search" | "refinement" | "comparison" | "clarification"
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

_VALID_INTENTS = frozenset({"search", "refinement", "comparison", "clarification"})

# Tokens that convey no actionable constraint on their own.
WEAK_TOKEN_SET: frozenset[str] = frozenset({
    "cheap", "cheapest", "best", "top", "lowest", "low", "high", "good",
    "better", "budget", "under", "over", "max", "min",
})

# Substrings that trigger comparison detection (checked on lowercased query).
_COMPARISON_PHRASES: tuple[str, ...] = (
    " vs ",
    " versus ",
    "compare",
    "comparison",
    "difference",
    "which is better",
)

# ---------------------------------------------------------------------------
# SIGNAL HELPERS (all pure, all O(n) or O(1))
# ---------------------------------------------------------------------------

def _has_signal(d: object) -> bool:
    """
    Returns True if `d` contains any meaningful slot signal:
      - category not None
      - price.min or price.max not None
      - any non-empty brand include/exclude list
      - any non-empty features required/excluded list
    Returns False on any structural problem (safe fallback).
    """
    if not isinstance(d, dict):
        return False
    try:
        if d.get("category") is not None:
            return True
        price = d.get("price") or {}
        if price.get("min") is not None or price.get("max") is not None:
            return True
        brand = d.get("brand") or {}
        if brand.get("include") or brand.get("exclude"):
            return True
        features = d.get("features") or {}
        if features.get("required") or features.get("excluded"):
            return True
    except Exception:
        pass
    return False


def _category_shift(slots: dict, state: dict) -> bool:
    """
    True when slots introduces a different, non-None category than state.
    Safe against missing keys.
    """
    try:
        new_cat  = slots.get("category")
        prev_cat = state.get("category")
        return (
            new_cat  is not None
            and prev_cat is not None
            and new_cat != prev_cat
        )
    except Exception:
        return False


def _is_weak_query(query_lower: str) -> bool:
    """
    A query is WEAK when:
      - it has ≤ 2 non-empty tokens, OR
      - every token belongs to WEAK_TOKEN_SET.
    Operates on an already-lowercased string.
    """
    tokens = [t for t in query_lower.split() if t]
    if not tokens:
        return True
    if len(tokens) <= 2:
        return True
    return all(t in WEAK_TOKEN_SET for t in tokens)


def _has_comparison_signal(query_lower: str, slots: dict) -> bool:
    """
    Returns True when ANY of the following hold:
      1. A comparison phrase appears in the lowercased query.
      2. slots.brand.include has ≥ 2 items.
      3. Query contains two tokens joined by "vs" or "and"
         (simple noun-pair heuristic; no regex backtracking).
    """
    # 1 — phrase match (single linear scan per phrase)
    for phrase in _COMPARISON_PHRASES:
        if phrase in query_lower:
            return True

    # 2 — multiple brands already selected
    try:
        if len(slots.get("brand", {}).get("include", [])) >= 2:
            return True
    except Exception:
        pass

    # 3 — "X vs Y" or "X and Y" two-token heuristic
    #     tokenise once; look for a join-word between two non-join tokens.
    try:
        tokens = [t for t in query_lower.split() if t]
        join_words = frozenset({"vs", "versus", "and"})
        for i, tok in enumerate(tokens):
            if tok in join_words and i > 0 and i < len(tokens) - 1:
                # both neighbours must be non-join-word, non-weak tokens
                left  = tokens[i - 1]
                right = tokens[i + 1]
                if left not in join_words and right not in join_words:
                    return True
    except Exception:
        pass

    return False


def _get_safe_str(d: object, key: str, default: str = "") -> str:
    """Safe dict string accessor; returns default on any problem."""
    try:
        val = d.get(key)  # type: ignore[union-attr]
        return val if isinstance(val, str) else default
    except Exception:
        return default


def _get_safe_list(d: object, *keys: str) -> list:
    """Safe nested list accessor. Returns [] on any problem."""
    try:
        cur: object = d
        for k in keys:
            cur = cur.get(k)  # type: ignore[union-attr]
            if cur is None:
                return []
        return cur if isinstance(cur, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def route_intent(slots: dict, state: dict, query: str) -> str:
    """
    Deterministic intent classifier.

    Returns one of: "search" | "refinement" | "comparison" | "clarification"
    Never raises; falls back to "search" on any unhandled error.

    Decision pipeline (evaluated in strict precedence order):
      1 — Comparison signal   → "comparison"
      2 — Category shift      → "search"
      3 — No context/signal   → "clarification"
      4 — Initial bootstrap   → "search"
      5 — Refinement (4 sub-cases)
      6 — Fallback            → "search"
    """
    try:
        # ── INPUT GUARD ───────────────────────────────────────────────────────
        if not isinstance(slots, dict):
            slots = {}
        if not isinstance(state, dict):
            state = {}
        if not isinstance(query, str):
            query = ""

        # ── PRECOMPUTE ALL SIGNALS ONCE ───────────────────────────────────────
        query_lower: str = query.lower()

        has_state:       bool = _has_signal(state)
        has_slots_signal: bool = _has_signal(slots)
        cat_shift:       bool = _category_shift(slots, state)
        weak_query:      bool = _is_weak_query(query_lower)
        comparison:      bool = _has_comparison_signal(query_lower, slots)

        slots_cat: str | None = slots.get("category") if isinstance(slots.get("category"), str) else None
        state_cat: str | None = state.get("category") if isinstance(state.get("category"), str) else None

        # ── STEP 1 — COMPARISON (HIGHEST PRIORITY) ────────────────────────────
        if comparison:
            return "comparison"

        # ── STEP 2 — HARD CATEGORY SHIFT ─────────────────────────────────────
        if cat_shift:
            return "search"

        # ── STEP 3 — CLARIFICATION (NO CONTEXT, NO SIGNAL) ───────────────────
        if not has_state and not has_slots_signal:
            return "clarification"

        # ── STEP 4 — INITIAL SEARCH (BOOTSTRAP) ──────────────────────────────
        if slots_cat is not None and state_cat is None:
            return "search"

        # ── STEP 5 — REFINEMENT (CONTEXT-AWARE) ──────────────────────────────
        if has_state:
            # 5A — Weak follow-up (e.g. "cheapest", "under 2k")
            if weak_query:
                return "refinement"

            # 5B — Partial slot update with no new category
            if slots_cat is None and has_slots_signal:
                return "refinement"

            # 5C — Same-category strengthening
            if slots_cat is not None and slots_cat == state_cat and has_slots_signal:
                return "refinement"

            # 5D — Negation-only update (exclusions present)
            brand_exclude   = _get_safe_list(slots, "brand", "exclude")
            feat_excluded   = _get_safe_list(slots, "features", "excluded")
            if brand_exclude or feat_excluded:
                return "refinement"

        # ── STEP 6 — FALLBACK ─────────────────────────────────────────────────
        return "search"

    except Exception:
        return "search"