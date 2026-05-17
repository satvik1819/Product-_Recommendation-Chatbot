"""
validator.py

Zero-trust normalization layer.
Sits between UNTRUSTED slot_extractor.py and the DETERMINISTIC pipeline.

Architecture: Slot Extractor → [THIS MODULE] → State Manager → Engine → Filters → Ranking
"""

import copy
import json
import logging
import re
import string

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH — EMPTY SCHEMA
# ---------------------------------------------------------------------------

EMPTY_SCHEMA: dict = {
    "category": None,
    "price": {"min": None, "max": None},
    "brand": {"include": [], "exclude": []},
    "features": {"required": [], "excluded": []},
    "intent_hint": "unknown",
}

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

_VALID_INTENT_HINTS = frozenset({"search", "refinement", "comparison", "unknown"})
_MAX_LIST_LEN = 20
_MAX_STR_LEN = 200
_PRICE_UPPER = 10_000_000

# Stage 5 — category normalization map (lowercase keys and values)
CATEGORY_MAP: dict[str, str] = {
    "earbuds": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "headset": "headphones",
    "bluetooth headset": "headphones",
    "tws earbuds": "headphones",
    "tws": "headphones",
    "in ear": "headphones",
    "over ear": "headphones",
    "iem": "headphones",
    "earbud": "headphones",
    "gaming headset": "headphones",
    "phone": "mobile",
    "phones": "mobile",
    "smartphone": "mobile",
    "smartphones": "mobile",
    "handset": "mobile",
    "mobiles": "mobile",
    "notebook": "laptop",
    "laptops": "laptop",
    "ultrabook": "laptop",
    "sneakers": "shoes",
    "footwear": "shoes",
    "trainers": "shoes",
    "running shoes": "shoes",
    "sports shoes": "shoes",
    "water bottle": "bottle",
    "flask": "bottle",
    "bottles": "bottle",
    "sipper": "bottle",
    "speakers": "speaker",
    "bluetooth speaker": "speaker",
    "portable speaker": "speaker",
    "soundbar": "speaker",
    "sound system": "speaker",
    "blankets": "blanket",
    "comforter": "blanket",
    "bedsheet": "blanket",
    "bedsheets": "blanket",
    "bed sheets": "blanket",
    "quilt": "blanket",
    "bed sheet": "blanket",
    "duvet": "blanket",
    "dslr": "camera",
    "mirrorless": "camera",
    "cameras": "camera",
    "webcam": "camera",
    "smartwatch": "watch",
    "fitness band": "watch",
    "watches": "watch",
    "wristwatch": "watch",
    "ipad": "tablet",
    "tablets": "tablet",
    "e-reader": "tablet",
    "keyboards": "keyboard",
    "mechanical keyboard": "keyboard",
    "mice": "mouse",
    "wireless mouse": "mouse",
    "gaming mouse": "mouse",
    "chargers": "charger",
    "power bank": "charger",
    "powerbank": "charger",
    "adapter": "charger",
    "backpack": "bag",
    "luggage": "bag",
    "suitcase": "bag",
    "bags": "bag",
    "rucksack": "bag",
    "t-shirt": "shirt",
    "tshirt": "shirt",
    "shirts": "shirt",
    "polo": "shirt",
    "trousers": "trouser",
    "pants": "trouser",
    "jeans": "trouser",
    "chinos": "trouser",
    "display": "monitor",
    "screen": "monitor",
    "monitors": "monitor",
    "tv": "television",
    "smart tv": "television",
    "led tv": "television",
    "oled": "television",
    "qled": "television",
    "fridge": "refrigerator",
    "freezer": "refrigerator",
    "washer": "washing machine",
    "front load": "washing machine",
    "top load": "washing machine",
    "shaver": "trimmer",
    "beard trimmer": "trimmer",
    "grooming": "trimmer",
    "shades": "sunglasses",
    "eyewear": "sunglasses",
    "sunglass": "sunglasses",
    "purse": "wallet",
    "cardholder": "wallet",
    "deodorant": "perfume",
    "cologne": "perfume",
    "deo": "perfume",
    "fragrance": "perfume",
}

# Stage 7 — feature noise tokens (not actionable constraints)
_NOISE_TOKENS: frozenset[str] = frozenset(
    {"good", "best", "cheap", "nice", "better", "top", "latest", "new"}
)

# ---------------------------------------------------------------------------
# SAFE MINIMAL SCHEMA BUILDER — never destroys category
# ---------------------------------------------------------------------------

def _safe_minimal(category: object = None) -> dict:
    """Return a minimal valid schema preserving category if available."""
    cat = None
    if category is not None:
        try:
            cat = _sanitize_string(category)
        except Exception:
            cat = None
        if cat is None and isinstance(category, str) and category.strip():
            cat = category.strip().lower()[:_MAX_STR_LEN]
    return {
        "category": cat,
        "price": {"min": None, "max": None},
        "brand": {"include": [], "exclude": []},
        "features": {"required": [], "excluded": []},
        "intent_hint": "search",
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _empty() -> dict:
    """Return a fresh deep copy of EMPTY_SCHEMA."""
    return copy.deepcopy(EMPTY_SCHEMA)


def _sanitize_string(value: object) -> str | None:
    """
    Stage 2 sanitizer (pure):
      - Accept str or safely coerce numeric types to str
      - lowercase
      - strip surrounding whitespace
      - strip leading/trailing punctuation
      - collapse internal runs of whitespace to a single space
      - truncate if length > _MAX_STR_LEN after cleaning
      - encode to ASCII, replacing non-ASCII with '' (safe for downstream)
    Returns None on empty result; never raises.
    """
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                return None

        cleaned = value.lower().strip()
        cleaned = cleaned.strip(string.punctuation)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
        cleaned = cleaned.strip()

        if not cleaned:
            return None

        if len(cleaned) > _MAX_STR_LEN:
            cleaned = cleaned[:_MAX_STR_LEN].strip()
            if not cleaned:
                return None

        return cleaned
    except Exception as exc:
        log.warning("[validator] _sanitize_string error for %r: %s", value, exc)
        return None


def parse_price(v: object) -> int | None:
    """
    Stage 3 — price parser with extended currency prefix support.

    Accepted inputs -> int result:
      int (non-bool, non-negative, <= _PRICE_UPPER)
      float that is_integer(), same bounds
      str "1000"  -> 1000
      str "1,000" -> 1000
      str "2k"    -> 2000
      str "2K"    -> 2000
      str "1000"  -> 1000
      str "$500"  -> 500

    Everything else -> None.
    """
    try:
        if isinstance(v, bool):
            return None

        if isinstance(v, int):
            if v < 0 or v > _PRICE_UPPER:
                return None
            return v

        if isinstance(v, float):
            if not v.is_integer():
                iv = round(v)
            else:
                iv = int(v)
            if iv < 0 or iv > _PRICE_UPPER:
                return None
            return iv

        if isinstance(v, str):
            s = v.strip()

            s = re.sub(r"^[₹$€£¥\s]+", "", s).strip()

            multiplier = 1
            if s.lower().endswith("k"):
                multiplier = 1000
                s = s[:-1].strip()
            elif s.lower().endswith("l") or s.lower().endswith("lac") or s.lower().endswith("lakh"):
                multiplier = 100_000
                s = re.sub(r"(?i)(lakh|lac|l)$", "", s).strip()

            s = s.replace(",", "").strip()

            if not re.fullmatch(r"\d+", s):
                digits = re.search(r"\d+", s)
                if digits:
                    s = digits.group()
                else:
                    return None

            try:
                result = int(s) * multiplier
            except (ValueError, OverflowError):
                return None

            if result < 0 or result > _PRICE_UPPER:
                return None
            return result

        return None
    except Exception as exc:
        log.warning("[validator] parse_price error for %r: %s", v, exc)
        return None


def _normalize_list(raw: object) -> list[str]:
    """
    Stage 4 — list normalization pipeline:
      1. Coerce bare str -> [str]; non-list/non-str -> []
      2. Sanitize each element (Stage 2 sanitizer)
      3. Drop None results
      4. Deduplicate (order-preserving)
      5. Truncate to _MAX_LIST_LEN
    """
    try:
        if isinstance(raw, str):
            raw = [raw]
        elif not isinstance(raw, list):
            if raw is None:
                raw = []
            else:
                try:
                    raw = list(raw)
                except Exception:
                    raw = []

        result: list[str] = []
        seen: set[str] = set()

        for item in raw:
            s = _sanitize_string(item)
            if s is not None and s not in seen:
                seen.add(s)
                result.append(s)
            if len(result) >= _MAX_LIST_LEN:
                break

        return result
    except Exception as exc:
        log.warning("[validator] _normalize_list error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def validate_slots(slots: dict) -> dict:
    """
    Zero-trust normalization of slot_extractor output.

    NEVER raises. ALWAYS returns a dict matching EMPTY_SCHEMA's exact shape.
    ALWAYS deterministic for identical inputs.
    NEVER destroys valid category or partial signals.

    Pipeline stages (in strict order):
      0  — Safe guard + deep copy
      1  — Shape coercion
      2  — String normalization
      3  — Price parsing
      4  — List normalization
      5  — Category normalization
      6  — Conflict resolution
      7  — Feature noise filter
      8  — Intent normalization
      9  — Minimum signal check (non-destructive)
      10 — Final assembly
      11 — Self-validation (repair, not reject)
      12 — Global safety wrapper (outermost try/except)
    """
    # Track the raw category early so we can always preserve it on any error path.
    _raw_category: object = None

    try:
        # ── STAGE 0 — SAFE GUARD + COPY ──────────────────────────────────────
        if not isinstance(slots, dict):
            if slots is None:
                log.info("[validator] slots is None — returning safe minimal schema")
            else:
                log.warning(
                    "[validator] slots is not a dict (got %s) — returning safe minimal schema",
                    type(slots),
                )
            return _safe_minimal()

        data: dict = copy.deepcopy(slots)

        # Capture raw category before any processing for error-path recovery.
        _raw_category = data.get("category")

        # ── STAGE 1 — SHAPE COERCION ─────────────────────────────────────────

        # category — accept str, int, or None; coerce others safely
        cat_raw = data.get("category")
        if cat_raw is None or isinstance(cat_raw, str):
            pass
        elif isinstance(cat_raw, (int, float)) and not isinstance(cat_raw, bool):
            data["category"] = str(cat_raw)
        else:
            data["category"] = None

        # price
        price = data.get("price")
        if isinstance(price, dict):
            p_min = price.get("min")
            p_max = price.get("max")
            if p_min is None and p_max is None:
                for alt_min in ("min_price", "price_min", "minimum", "from"):
                    if alt_min in data and data[alt_min] is not None:
                        p_min = data[alt_min]
                        break
                for alt_max in ("max_price", "price_max", "maximum", "to", "under", "below"):
                    if alt_max in data and data[alt_max] is not None:
                        p_max = data[alt_max]
                        break
            data["price"] = {"min": p_min, "max": p_max}
        elif price is None:
            p_min = None
            p_max = None
            for alt_min in ("min_price", "price_min", "minimum", "from"):
                if alt_min in data and data[alt_min] is not None:
                    p_min = data[alt_min]
                    break
            for alt_max in ("max_price", "price_max", "maximum", "to", "under", "below"):
                if alt_max in data and data[alt_max] is not None:
                    p_max = data[alt_max]
                    break
            data["price"] = {"min": p_min, "max": p_max}
        else:
            data["price"] = {"min": None, "max": None}

        # brand
        brand = data.get("brand")
        if isinstance(brand, dict):
            inc = brand.get("include")
            exc = brand.get("exclude")
            data["brand"] = {
                "include": inc if isinstance(inc, (list, str)) else ([] if inc is None else [str(inc)]),
                "exclude": exc if isinstance(exc, (list, str)) else ([] if exc is None else [str(exc)]),
            }
        elif isinstance(brand, str) and brand.strip():
            data["brand"] = {"include": [brand.strip()], "exclude": []}
        elif isinstance(brand, list):
            data["brand"] = {"include": brand, "exclude": []}
        else:
            data["brand"] = {"include": [], "exclude": []}

        # features
        features = data.get("features")
        if isinstance(features, dict):
            req = features.get("required")
            excl = features.get("excluded")
            data["features"] = {
                "required": req if isinstance(req, (list, str)) else ([] if req is None else []),
                "excluded": excl if isinstance(excl, (list, str)) else ([] if excl is None else []),
            }
        elif isinstance(features, list):
            data["features"] = {"required": features, "excluded": []}
        else:
            data["features"] = {"required": [], "excluded": []}

        # intent_hint
        if not isinstance(data.get("intent_hint"), str):
            data["intent_hint"] = "search"

        # ── STAGE 2 — STRING NORMALIZATION ───────────────────────────────────

        if data["category"] is not None:
            sanitized_cat = _sanitize_string(data["category"])
            if sanitized_cat is not None:
                data["category"] = sanitized_cat
            else:
                # Sanitizer returned None (e.g. non-ASCII only) — keep raw stripped lowercase.
                fallback_cat = str(data["category"]).lower().strip()[:_MAX_STR_LEN]
                data["category"] = fallback_cat if fallback_cat else None

        data["intent_hint"] = _sanitize_string(data["intent_hint"]) or "search"

        # ── STAGE 3 — PRICE PARSING ───────────────────────────────────────────
        data["price"]["min"] = parse_price(data["price"]["min"])
        data["price"]["max"] = parse_price(data["price"]["max"])

        # ── STAGE 4 — LIST NORMALIZATION ─────────────────────────────────────
        data["brand"]["include"]     = _normalize_list(data["brand"]["include"])
        data["brand"]["exclude"]     = _normalize_list(data["brand"]["exclude"])
        data["features"]["required"] = _normalize_list(data["features"]["required"])
        data["features"]["excluded"] = _normalize_list(data["features"]["excluded"])

        # ── STAGE 5 — CATEGORY NORMALIZATION ─────────────────────────────────
        cat = data["category"]
        if cat is not None:
            if cat in CATEGORY_MAP:
                cat = CATEGORY_MAP[cat]
            else:
                for key, mapped in sorted(CATEGORY_MAP.items(), key=lambda x: -len(x[0])):
                    if key in cat:
                        cat = mapped
                        break
            # Always preserve — even if no map hit, keep original cleaned string.
            data["category"] = cat

        # ── STAGE 6 — CONFLICT RESOLUTION ────────────────────────────────────

        p_min = data["price"]["min"]
        p_max = data["price"]["max"]
        if p_min is not None and p_max is not None and p_min > p_max:
            log.info("[validator] price range inverted (%d > %d) — swapping", p_min, p_max)
            data["price"]["min"], data["price"]["max"] = p_max, p_min

        b_inc = set(data["brand"]["include"])
        b_exc = set(data["brand"]["exclude"])
        overlap_b = b_inc & b_exc
        if overlap_b:
            log.info("[validator] brand include/exclude overlap %s — removing from include", overlap_b)
            data["brand"]["include"] = [x for x in data["brand"]["include"] if x not in overlap_b]

        f_req  = set(data["features"]["required"])
        f_excl = set(data["features"]["excluded"])
        overlap_f = f_req & f_excl
        if overlap_f:
            log.info("[validator] feature required/excluded overlap %s — removing from required", overlap_f)
            data["features"]["required"] = [x for x in data["features"]["required"] if x not in overlap_f]

        # ── STAGE 7 — FEATURE NOISE FILTER ───────────────────────────────────
        data["features"]["required"] = [
            t for t in data["features"]["required"] if t not in _NOISE_TOKENS
        ]
        data["features"]["excluded"] = [
            t for t in data["features"]["excluded"] if t not in _NOISE_TOKENS
        ]

        # ── STAGE 8 — INTENT NORMALIZATION ───────────────────────────────────
        intent = data["intent_hint"]
        if intent not in _VALID_INTENT_HINTS:
            intent = "search"
        # Promote "unknown" -> "search" to keep the pipeline flowing.
        if intent == "unknown":
            intent = "search"
        data["intent_hint"] = intent

        # ── STAGE 9 — MINIMUM SIGNAL CHECK (NON-DESTRUCTIVE) ─────────────────
        has_signal = (
            data["category"] is not None
            or data["price"]["min"] is not None
            or data["price"]["max"] is not None
            or bool(data["brand"]["include"])
            or bool(data["brand"]["exclude"])
            or bool(data["features"]["required"])
            or bool(data["features"]["excluded"])
        )
        if not has_signal:
            log.warning("[validator] weak signal — preserving minimal structure")
            return {
                "category": data.get("category"),
                "price": {"min": None, "max": None},
                "brand": {"include": [], "exclude": []},
                "features": {"required": [], "excluded": []},
                "intent_hint": "search",
            }

        # ── STAGE 10 — FINAL ASSEMBLY ─────────────────────────────────────────
        output: dict = {
            "category": data["category"],
            "price": {
                "min": data["price"]["min"],
                "max": data["price"]["max"],
            },
            "brand": {
                "include": data["brand"]["include"],
                "exclude": data["brand"]["exclude"],
            },
            "features": {
                "required": data["features"]["required"],
                "excluded": data["features"]["excluded"],
            },
            "intent_hint": data["intent_hint"],
        }

        # ── STAGE 11 — SELF-VALIDATION (REPAIR, NOT REJECT) ──────────────────
        _required_keys = {"category", "price", "brand", "features", "intent_hint"}

        def _repair_and_validate() -> None:
            # Key set — add missing keys with safe defaults.
            for k in _required_keys:
                if k not in output:
                    log.warning("[validator] self-validation: missing key %r — adding default", k)
                    output[k] = copy.deepcopy(EMPTY_SCHEMA.get(k))

            # category
            cat_out = output.get("category")
            if cat_out is not None and not isinstance(cat_out, str):
                output["category"] = str(cat_out).strip().lower()[:_MAX_STR_LEN] or None
            if isinstance(output.get("category"), str) and not output["category"]:
                output["category"] = None

            # price
            pr = output.get("price")
            if not isinstance(pr, dict) or set(pr.keys()) != {"min", "max"}:
                log.warning("[validator] self-validation: price shape invalid — resetting")
                output["price"] = {"min": None, "max": None}
            else:
                for pk in ("min", "max"):
                    pv = pr.get(pk)
                    if pv is None:
                        continue
                    if isinstance(pv, bool) or not isinstance(pv, int):
                        log.warning("[validator] self-validation: price.%s not int (%r) — clearing", pk, pv)
                        output["price"][pk] = None
                    elif pv < 0 or pv > _PRICE_UPPER:
                        log.warning("[validator] self-validation: price.%s out of bounds (%d) — clearing", pk, pv)
                        output["price"][pk] = None
                if (
                    output["price"]["min"] is not None
                    and output["price"]["max"] is not None
                    and output["price"]["min"] > output["price"]["max"]
                ):
                    output["price"]["min"], output["price"]["max"] = (
                        output["price"]["max"],
                        output["price"]["min"],
                    )
                    log.info("[validator] self-validation swapped inverted price range")

            # brand
            br = output.get("brand")
            if not isinstance(br, dict) or set(br.keys()) != {"include", "exclude"}:
                log.warning("[validator] self-validation: brand shape invalid — resetting")
                output["brand"] = {"include": [], "exclude": []}
            else:
                for bk in ("include", "exclude"):
                    lst = br.get(bk)
                    if not isinstance(lst, list):
                        log.warning("[validator] self-validation: brand.%s not list — resetting", bk)
                        output["brand"][bk] = []
                    else:
                        output["brand"][bk] = [
                            x for x in lst if isinstance(x, str) and x
                        ][:_MAX_LIST_LEN]
                overlap = set(output["brand"]["include"]) & set(output["brand"]["exclude"])
                if overlap:
                    output["brand"]["include"] = [
                        x for x in output["brand"]["include"] if x not in overlap
                    ]
                    log.info("[validator] self-validation removed brand overlap: %s", overlap)

            # features
            ft = output.get("features")
            if not isinstance(ft, dict) or set(ft.keys()) != {"required", "excluded"}:
                log.warning("[validator] self-validation: features shape invalid — resetting")
                output["features"] = {"required": [], "excluded": []}
            else:
                for fk in ("required", "excluded"):
                    lst = ft.get(fk)
                    if not isinstance(lst, list):
                        log.warning("[validator] self-validation: features.%s not list — resetting", fk)
                        output["features"][fk] = []
                    else:
                        output["features"][fk] = [
                            x for x in lst if isinstance(x, str) and x
                        ][:_MAX_LIST_LEN]
                overlap_f2 = set(output["features"]["required"]) & set(output["features"]["excluded"])
                if overlap_f2:
                    output["features"]["required"] = [
                        x for x in output["features"]["required"] if x not in overlap_f2
                    ]
                    log.info("[validator] self-validation removed feature overlap: %s", overlap_f2)

            # intent_hint
            if output.get("intent_hint") not in _VALID_INTENT_HINTS:
                output["intent_hint"] = "search"
            if output.get("intent_hint") == "unknown":
                output["intent_hint"] = "search"

            # JSON-serializability — last resort strip on failure.
            try:
                json.dumps(output)
            except (TypeError, ValueError, OverflowError) as exc:
                log.warning("[validator] self-validation: json-serialization failed (%s) — stripping lists", exc)
                output["brand"] = {"include": [], "exclude": []}
                output["features"] = {"required": [], "excluded": []}

        _repair_and_validate()

        # ── STAGE 12 — RETURN (deep copy prevents caller mutation) ────────────
        return copy.deepcopy(output)

    except Exception as exc:
        log.warning(
            "[validator] validate_slots unhandled exception: %s — returning safe minimal schema", exc
        )
        return _safe_minimal(_raw_category)