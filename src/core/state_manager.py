"""
state_manager.py

Conversational state accumulator.
Merges validated slot updates into persistent, contradiction-free session state.

Architecture: Validator → [THIS MODULE] → Engine → Filters → Ranking
"""

import copy
import json

# ---------------------------------------------------------------------------
# EMPTY SCHEMA — single source of truth for blank state
# ---------------------------------------------------------------------------

EMPTY_SCHEMA: dict = {
    "category": None,
    "price": {"min": None, "max": None},
    "brand": {"include": [], "exclude": []},
    "features": {"required": [], "excluded": []},
    "intent_hint": "unknown",
}

_VALID_INTENT_HINTS = frozenset({"search", "refinement", "comparison", "unknown"})
_MAX_LIST_LEN = 20

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _empty() -> dict:
    """Return a fresh deep copy of EMPTY_SCHEMA."""
    return copy.deepcopy(EMPTY_SCHEMA)


def _is_valid_state(s: object) -> bool:
    """
    Structural guard: confirm s has the exact schema shape and correct types.
    Does NOT enforce business rules (those belong in validator.py).
    Returns False on any deviation so callers can fall back safely.
    """
    if not isinstance(s, dict):
        return False

    required_keys = {"category", "price", "brand", "features", "intent_hint"}
    if set(s.keys()) != required_keys:
        return False

    # category
    cat = s.get("category")
    if cat is not None and not isinstance(cat, str):
        return False

    # price
    price = s.get("price")
    if not isinstance(price, dict) or set(price.keys()) != {"min", "max"}:
        return False
    for pv in (price["min"], price["max"]):
        if pv is None:
            continue
        if isinstance(pv, bool) or not isinstance(pv, int):
            return False

    # brand
    brand = s.get("brand")
    if not isinstance(brand, dict) or set(brand.keys()) != {"include", "exclude"}:
        return False
    for lst in (brand["include"], brand["exclude"]):
        if not isinstance(lst, list):
            return False
        if not all(isinstance(x, str) for x in lst):
            return False

    # features
    features = s.get("features")
    if not isinstance(features, dict) or set(features.keys()) != {"required", "excluded"}:
        return False
    for lst in (features["required"], features["excluded"]):
        if not isinstance(lst, list):
            return False
        if not all(isinstance(x, str) for x in lst):
            return False

    # intent_hint
    if not isinstance(s.get("intent_hint"), str):
        return False

    return True


def _is_empty_schema(s: dict) -> bool:
    """
    Returns True if s is functionally equivalent to EMPTY_SCHEMA
    (no real signal in any field).
    """
    return (
        s.get("category") is None
        and s.get("price", {}).get("min") is None
        and s.get("price", {}).get("max") is None
        and not s.get("brand", {}).get("include")
        and not s.get("brand", {}).get("exclude")
        and not s.get("features", {}).get("required")
        and not s.get("features", {}).get("excluded")
        and s.get("intent_hint", "unknown") == "unknown"
    )


def _ordered_union(a: list[str], b: list[str]) -> list[str]:
    """
    Order-preserving union of two string lists.
    Items from `a` appear first; items from `b` are appended if not already present.
    Result is capped at _MAX_LIST_LEN.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in a + b:
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= _MAX_LIST_LEN:
            break
    return result


def _resolve_conflicts(include: list[str], exclude: list[str]) -> list[str]:
    """
    Remove any item from `include` that also appears in `exclude`.
    Exclude always wins. Returns the filtered include list.
    """
    exc_set = set(exclude)
    return [x for x in include if x not in exc_set]


def _is_valid_final(output: dict) -> bool:
    """
    Post-merge sanity check.  Enforces both structural validity AND
    the subset of business invariants that merge_state is responsible for:
      - no duplicates within any list
      - include ∩ exclude = ∅ for brand and features
      - price.min ≤ price.max when both present
      - JSON-serializable
    """
    if not _is_valid_state(output):
        return False

    # no duplicates
    for path in (
        output["brand"]["include"],
        output["brand"]["exclude"],
        output["features"]["required"],
        output["features"]["excluded"],
    ):
        if len(path) != len(set(path)):
            return False

    # no include/exclude intersection
    if set(output["brand"]["include"]) & set(output["brand"]["exclude"]):
        return False
    if set(output["features"]["required"]) & set(output["features"]["excluded"]):
        return False

    # price ordering
    p = output["price"]
    if p["min"] is not None and p["max"] is not None and p["min"] > p["max"]:
        return False

    # JSON-serializable
    try:
        json.dumps(output)
    except (TypeError, ValueError, OverflowError):
        return False

    return True


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def merge_state(prev_state: dict, new_slots: dict) -> dict:
    """
    Merge a validated slot update into persistent conversational state.

    Guarantees:
      - NEVER raises
      - NEVER mutates inputs
      - ALWAYS returns a structurally valid dict matching EMPTY_SCHEMA's shape
      - ALWAYS deterministic for identical inputs

    Steps (in strict order):
      0  — Safe guard + deep copy
      1  — Category handling (possible hard reset)
      2  — Price merge (new overwrites old per field, None = keep)
      3  — Brand merge (union; exclude wins on conflict)
      4  — Features merge (union; excluded wins on conflict)
      5  — Negation priority (new exclusions evict previous includes)
      6  — Intent update (new wins unless "unknown")
      7  — Empty update short-circuit
      8  — Final sanity check
      9  — Global safety wrapper (outermost try/except)
    """
    try:
        # ── STEP 0 — SAFE GUARD + COPY ────────────────────────────────────────
        if not _is_valid_state(prev_state):
            prev_state = _empty()

        if not _is_valid_state(new_slots):
            # new_slots is unusable — return prev unchanged (deep copy)
            return copy.deepcopy(prev_state)

        # Work on independent copies; never touch caller-owned objects.
        prev = copy.deepcopy(prev_state)
        new  = copy.deepcopy(new_slots)

        # ── STEP 7 — EMPTY UPDATE SHORT-CIRCUIT ──────────────────────────────
        # Checked early (before mutation) so we can return prev intact.
        if _is_empty_schema(new):
            return prev

        # ── STEP 1 — CATEGORY HANDLING ───────────────────────────────────────
        new_cat  = new.get("category")
        prev_cat = prev.get("category")

        if new_cat is not None:
            if prev_cat is None:
                # First time a category is set — accept it.
                prev["category"] = new_cat

            elif new_cat != prev_cat:
                # Category changed → HARD RESET: discard all accumulated state
                # except the new category.
                prev = _empty()
                prev["category"] = new_cat
                # Carry over the new intent (will be applied in Step 6).
                # All other fields start clean — skip to Step 6.
                prev["intent_hint"] = (
                    new["intent_hint"]
                    if new["intent_hint"] != "unknown"
                    else "unknown"
                )
                # After a hard reset, the only other fields that can come from
                # new_slots are price/brand/features — fall through to merge them
                # against the now-empty prev.

            # else: same category — keep everything, continue merging below.

        # ── STEP 2 — PRICE MERGE ─────────────────────────────────────────────
        if new["price"]["min"] is not None:
            prev["price"]["min"] = new["price"]["min"]
        # else: keep prev value

        if new["price"]["max"] is not None:
            prev["price"]["max"] = new["price"]["max"]
        # else: keep prev value

        # Enforce min ≤ max after merge; swap rather than discard.
        p_min = prev["price"]["min"]
        p_max = prev["price"]["max"]
        if p_min is not None and p_max is not None and p_min > p_max:
            prev["price"]["min"], prev["price"]["max"] = p_max, p_min

        # ── STEP 3 — BRAND MERGE ─────────────────────────────────────────────
        merged_b_inc = _ordered_union(prev["brand"]["include"], new["brand"]["include"])
        merged_b_exc = _ordered_union(prev["brand"]["exclude"], new["brand"]["exclude"])

        # ── STEP 4 — FEATURES MERGE ──────────────────────────────────────────
        merged_f_req  = _ordered_union(prev["features"]["required"],  new["features"]["required"])
        merged_f_excl = _ordered_union(prev["features"]["excluded"], new["features"]["excluded"])

        # ── STEP 5 — NEGATION PRIORITY ────────────────────────────────────────
        # New exclusions must evict anything in the (newly merged) include sets.
        # Applied AFTER both lists have been unioned so that a token introduced
        # in new_slots.include AND new_slots.exclude in the same turn is also
        # resolved correctly (excluded wins).
        merged_b_inc  = _resolve_conflicts(merged_b_inc,  merged_b_exc)
        merged_f_req  = _resolve_conflicts(merged_f_req,  merged_f_excl)

        prev["brand"]["include"]    = merged_b_inc
        prev["brand"]["exclude"]    = merged_b_exc
        prev["features"]["required"] = merged_f_req
        prev["features"]["excluded"] = merged_f_excl

        # ── STEP 6 — INTENT UPDATE ────────────────────────────────────────────
        if new["intent_hint"] != "unknown":
            prev["intent_hint"] = new["intent_hint"]
        # else: keep previous intent

        # ── STEP 8 — FINAL SANITY ────────────────────────────────────────────
        if not _is_valid_final(prev):
            return copy.deepcopy(prev_state)

        return copy.deepcopy(prev)

    except Exception:
        # ── STEP 9 — GLOBAL SAFETY ───────────────────────────────────────────
        # Any unhandled exception: return the last known-good state.
        try:
            return copy.deepcopy(prev_state)
        except Exception:
            return _empty()