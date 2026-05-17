from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GENERIC_KEYWORDS = frozenset({
    "men", "women", "best", "cheap", "top", "good", "great",
    "nice", "affordable", "budget", "premium", "latest", "new",
    "buy", "get", "find", "for", "under",
})

GENERIC_CATEGORIES = frozenset({
    "electronics", "items", "products", "things", "goods", "stuff",
    "accessories", "gadgets", "devices",
})

_NORMALIZED_ATTR = "_filters_normalized"
NORMALIZED_CACHE: Optional[pd.DataFrame] = None
CACHE_ID: Optional[int] = None
MASK_CACHE: dict[str, pd.Series] = {}

_CATEGORY_CANONICAL_ALIASES: dict[str, str] = {}


def _normalize_category(category: str) -> str:
    if not category:
        return ""
    return re.sub(r"[^a-z0-9\s]", "", str(category).lower()).strip()


def _register_category_group(canonical: str, aliases: list[str]) -> None:
    c = _normalize_category(canonical)
    if not c:
        return
    _CATEGORY_CANONICAL_ALIASES[c] = c
    for a in aliases:
        k = _normalize_category(a)
        if k:
            _CATEGORY_CANONICAL_ALIASES[k] = c


def _init_category_map() -> None:
    if _CATEGORY_CANONICAL_ALIASES:
        return
    _register_category_group("headphones", [
        "headphone", "headphones", "earphones", "earbuds", "earbud", "headset",
        "earphone", "tws", "iem", "in ear", "gaming headset",
    ])
    _register_category_group("mobile", [
        "mobile", "mobiles", "phone", "phones", "smartphone", "smartphones", "handset",
    ])
    _register_category_group("laptop", ["laptop", "notebook", "laptops", "ultrabook", "chromebook", "macbook"])
    _register_category_group("shoes", [
        "shoes", "sneakers", "footwear", "shoe", "trainers", "sports shoes", "running shoes",
    ])
    _register_category_group("bottle", ["bottle", "water bottle", "flask", "bottles", "sipper"])
    _register_category_group("speaker", [
        "speaker", "speakers", "bluetooth speaker", "portable speaker",
        "sound system", "soundbar",
    ])
    _register_category_group("blanket", [
        "blanket", "blankets", "comforter", "bedsheet", "quilt", "bed sheet", "duvet",
    ])
    _register_category_group("camera", ["camera", "dslr", "mirrorless", "cameras", "webcam"])
    _register_category_group("watch", [
        "watch", "smartwatch", "fitness band", "watches", "wristwatch",
    ])
    _register_category_group("tablet", ["tablet", "ipad", "tablets", "e-reader"])
    _register_category_group("keyboard", ["keyboard", "keyboards", "mechanical keyboard"])
    _register_category_group("mouse", ["mouse", "mice", "wireless mouse", "gaming mouse"])
    _register_category_group("charger", [
        "charger", "chargers", "power bank", "adapter", "powerbank",
    ])
    _register_category_group("bag", ["bag", "backpack", "luggage", "suitcase", "bags", "rucksack"])
    _register_category_group("shirt", ["shirt", "shirts", "t-shirt", "tshirt", "polo"])
    _register_category_group("trouser", ["trouser", "trousers", "pants", "jeans", "chinos"])
    _register_category_group("monitor", ["monitor", "display", "screen", "monitors"])
    _register_category_group("television", [
        "television", "tv", "smart tv", "led tv", "oled", "qled",
    ])
    _register_category_group("refrigerator", ["refrigerator", "fridge", "freezer"])
    _register_category_group("washing machine", [
        "washing machine", "washer", "front load", "top load",
    ])
    _register_category_group("trimmer", ["trimmer", "shaver", "beard trimmer", "grooming"])
    _register_category_group("sunglasses", ["sunglasses", "shades", "eyewear", "sunglass"])
    _register_category_group("wallet", ["wallet", "purse", "cardholder"])
    _register_category_group("perfume", [
        "perfume", "deodorant", "cologne", "deo", "fragrance",
    ])
    _register_category_group("pen", [
        "pen", "pens", "gel pen", "ball pen", "ballpoint", "stationery",
    ])


_init_category_map()

RELATED_CANONICAL_GROUPS: dict[str, frozenset[str]] = {
    "laptop": frozenset({"laptop", "tablet", "keyboard", "mouse", "monitor"}),
    "mobile": frozenset({"mobile", "tablet"}),
    "tablet": frozenset({"tablet", "laptop", "mobile"}),
    "headphones": frozenset({"headphones", "speaker"}),
    "speaker": frozenset({"speaker", "headphones"}),
    "watch": frozenset({"watch"}),
    "camera": frozenset({"camera"}),
    "television": frozenset({"television"}),
    "perfume": frozenset({"perfume"}),
    "pen": frozenset({"pen"}),
    "shoes": frozenset({"shoes"}),
    "shirt": frozenset({"shirt", "trouser"}),
    "trouser": frozenset({"trouser", "shirt"}),
    "charger": frozenset({"charger", "mobile"}),
    "bag": frozenset({"bag"}),
    "blanket": frozenset({"blanket"}),
    "bottle": frozenset({"bottle"}),
}


def resolve_query_category(category: Optional[str]) -> str:
    return _resolve_canonical_category(category)


def get_related_canonicals(canonical: str | None) -> list[str]:
    if not canonical:
        return []
    c = _resolve_canonical_category(canonical)
    if not c:
        return []
    group = RELATED_CANONICAL_GROUPS.get(c)
    if group:
        return sorted(group)
    return [c]


def _resolve_canonical_category(category: Optional[str]) -> str:
    if not category:
        return ""
    raw = _normalize_category(category)
    if not raw:
        return ""
    return _CATEGORY_CANONICAL_ALIASES.get(raw, raw)


def _canonicalize_text_tokens(text: str) -> set[str]:
    try:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return {_CATEGORY_CANONICAL_ALIASES.get(t, t) for t in tokens if t}
    except Exception:
        return set()


def _strict_brand_match(row: pd.Series, brand: str) -> bool:
    if not brand:
        return True
    try:
        brand = str(brand).lower().strip()
        row_brand = str(row.get("brand", "")).lower()
        return bool(re.search(rf"\b{re.escape(brand)}\b", row_brand))
    except Exception:
        return False


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _sanitize_indices(filtered_df: pd.DataFrame, valid_indices: set) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in filtered_df.index.tolist():
        if i in valid_indices and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _pct(before: int, after: int) -> str:
    if before == 0:
        return "N/A"
    drop = 100.0 * (before - after) / before
    arrow = "↓" if after <= before else "↑"
    return f"{arrow} {drop:.1f}%"


class StructuredQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query:     Optional[str]       = None
    max_price: Optional[int]       = None
    min_price: Optional[int]       = None
    category:  Optional[str]       = None
    brand:     Optional[str]       = None
    keywords:  Optional[list[str]] = None
    category_allowlist: Optional[list[str]] = None

    @field_validator("max_price", "min_price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        if v is None:
            return None
        try:
            price = int(float(str(v).replace(",", "").replace("₹", "").strip()))
            return price if price > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("category", "brand", "query", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        return s if s else None

    @field_validator("keywords", mode="before")
    @classmethod
    def coerce_keywords(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            return None
        try:
            cleaned = [
                str(k).strip().lower()
                for k in v
                if str(k).strip() and str(k).strip().lower() not in GENERIC_KEYWORDS
            ]
            return cleaned if cleaned else None
        except Exception:
            return None

    @field_validator("category_allowlist", mode="before")
    @classmethod
    def coerce_allowlist(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            return None
        out: list[str] = []
        for x in v:
            c = _resolve_canonical_category(str(x).strip())
            if c and c not in out:
                out.append(c)
        return out if out else None

    @model_validator(mode="after")
    def check_price_range(self) -> "StructuredQuery":
        try:
            if (
                self.min_price is not None
                and self.max_price is not None
                and self.min_price > self.max_price
            ):
                self.min_price, self.max_price = self.max_price, self.min_price
        except Exception:
            self.min_price = None
            self.max_price = None
        return self


def validate_query(raw: dict) -> StructuredQuery:
    if not isinstance(raw, dict):
        log.warning("validate_query received non-dict input (%s) — using empty query.", type(raw))
        return StructuredQuery()

    try:
        sq = StructuredQuery.model_validate(raw)
    except Exception as exc:
        log.warning("StructuredQuery validation error (%s) — using empty query.", exc)
        sq = StructuredQuery()

    if sq.category and sq.category.lower().strip() in GENERIC_CATEGORIES:
        inferred = None
        if sq.query:
            tokens = re.findall(r"[a-z0-9]+", str(sq.query).lower())
            for token in tokens:
                resolved = _CATEGORY_CANONICAL_ALIASES.get(token)
                if resolved and resolved not in GENERIC_CATEGORIES:
                    inferred = resolved
                    break
        sq = sq.model_copy(update={"category": inferred})

    log.info(
        "Parsed query | query=%r | max_price=%s | min_price=%s | "
        "category=%s | brand=%s | keywords=%s | allowlist=%s",
        sq.query, sq.max_price, sq.min_price, sq.category, sq.brand, sq.keywords,
        sq.category_allowlist,
    )
    return sq


def normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df, _NORMALIZED_ATTR, False):
        return df

    for col in ["brand", "category", "main_category", "sub_category"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    df["brand"] = df["brand"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["main_category"] = df["main_category"].astype(str).str.lower().str.strip()
    df["sub_category"] = df["sub_category"].astype(str).str.lower().str.strip()

    if "final_price" in df.columns:
        df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")

    if "normalized_text" not in df.columns:
        combined = (
            df["category"].astype(str)
            + " "
            + df["main_category"].astype(str)
            + " "
            + df["sub_category"].astype(str)
        ).astype(str).str.lower()
        combined = combined.str.replace(r"[^a-z0-9\s]", " ", regex=True)
        combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
        combined = combined.str.split().str.join(" ")
        df["normalized_text"] = combined

    if "canonical_tokens" not in df.columns:
        df["canonical_tokens"] = df["normalized_text"].map(_canonicalize_text_tokens)
        df["canonical_tokens"] = df["canonical_tokens"].map(frozenset)

    try:
        object.__setattr__(df, _NORMALIZED_ATTR, True)
    except AttributeError:
        pass

    return df


def _defensive_drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        if "final_price" not in df.columns:
            return df
        before = len(df)
        df = df[df["final_price"].notna()].copy()
        if before != len(df):
            log.info("[filters] defensive_drop: removed %d rows with missing price", before - len(df))
        return df
    except Exception:
        return df


def _get_or_build_canonical_mask(df: pd.DataFrame, canonical: str) -> pd.Series:
    cache_key = f"{id(df)}:{canonical}"
    if cache_key in MASK_CACHE:
        cached = MASK_CACHE[cache_key]
        if len(cached) == len(df) and cached.index.equals(df.index):
            return cached

    try:
        if "canonical_tokens" not in df.columns:
            df = normalize_inputs(df)
        mask = pd.Series(
            [canonical in t for t in df["canonical_tokens"]],
            index=df.index,
        )
    except Exception:
        mask = pd.Series(False, index=df.index)

    if len(MASK_CACHE) < 100:
        MASK_CACHE[cache_key] = mask

    return mask


def apply_category_allowlist_filter(df: pd.DataFrame, allowlist: list[str]) -> pd.DataFrame:
    if not allowlist:
        return _empty_like(df)
    if "canonical_tokens" not in df.columns:
        df = normalize_inputs(df)
    mask = pd.Series(False, index=df.index)
    for c in allowlist:
        mask |= _get_or_build_canonical_mask(df, c)
    result = df.loc[mask]
    log.info("[filters] category_allowlist | rows=%d list=%s", len(result), allowlist)
    return result


def apply_category_filter(df: pd.DataFrame, sq: StructuredQuery) -> pd.DataFrame:
    if sq.category is None:
        return df

    try:
        canonical = _resolve_canonical_category(sq.category)
        if not canonical:
            return _empty_like(df)

        before = len(df)
        if "canonical_tokens" not in df.columns:
            df = normalize_inputs(df)

        mask = _get_or_build_canonical_mask(df, canonical)
        result = df.loc[mask]
        log.info(
            "[filters] after_category | rows=%d (%s) canonical=%r",
            len(result), _pct(before, len(result)), canonical,
        )

        if len(result) == 0 and "normalized_text" in df.columns:
            relax_mask = df["normalized_text"].str.contains(
                re.escape(canonical), na=False
            )
            result = df.loc[relax_mask]

        if len(result) == 0:
            return result

        return result
    except Exception:
        return _empty_like(df)


def apply_brand_filter(
    df:      pd.DataFrame,
    sq:      StructuredQuery,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    if not sq.brand:
        return df

    try:
        if "brand" not in df.columns:
            return _empty_like(df)

        before = len(df)
        b = str(sq.brand).lower().strip()
        brand_col = df["brand"].astype(str).str.lower()

        mask = brand_col.str.contains(
            rf"\b{re.escape(b)}\b",
            regex=True,
            na=False,
        )
        result = df.loc[mask]

        if len(result) == 0:
            mask_sub = brand_col.str.contains(re.escape(b), na=False)
            result = df.loc[mask_sub]

        if len(result) == 0:
            return result

        log.info(
            "[filters] after_brand | rows=%d (%s) brand=%r",
            len(result), _pct(before, len(result)), b,
        )
        return result
    except Exception:
        return _empty_like(df)


def apply_price_filter(df: pd.DataFrame, sq: StructuredQuery) -> pd.DataFrame:
    if sq.max_price is None and sq.min_price is None:
        return df

    try:
        if "final_price" not in df.columns:
            return _empty_like(df)

        before = len(df)
        mask = pd.Series(True, index=df.index)
        if sq.min_price is not None:
            mask &= df["final_price"] >= sq.min_price
        if sq.max_price is not None:
            mask &= df["final_price"] <= sq.max_price

        result = df[mask]
        log.info(
            "[filters] after_price | rows=%d (%s) min=%s max=%s",
            len(result),
            _pct(before, len(result)),
            sq.min_price,
            sq.max_price,
        )

        if len(result) == 0:
            return result

        return result
    except Exception:
        return _empty_like(df)


def apply_keyword_scoring(df: pd.DataFrame, sq: StructuredQuery) -> pd.DataFrame:
    return df


def apply_keyword_filter(
    df:       pd.DataFrame,
    sq:      StructuredQuery,
    min_rows: int = 200,
) -> pd.DataFrame:
    return df


def apply_candidate_control(df: pd.DataFrame, sq: StructuredQuery) -> pd.DataFrame:
    return df


def fallback_strategy(
    full_df: pd.DataFrame,
    sq:      StructuredQuery,
    current: pd.DataFrame,
) -> pd.DataFrame:
    return current


def _apply_category_stage(working: pd.DataFrame, sq: StructuredQuery) -> pd.DataFrame:
    if sq.category_allowlist:
        return apply_category_allowlist_filter(working, sq.category_allowlist)
    if sq.category:
        return apply_category_filter(working, sq)
    return working


def filter_products(
    df:        pd.DataFrame,
    raw_query: dict,
) -> tuple[list[int], pd.DataFrame]:
    global NORMALIZED_CACHE, CACHE_ID
    total = len(df)
    log.info("[filters] total_rows=%d | raw_query=%s", total, raw_query)

    if total == 0:
        return [], _empty_like(df)

    valid_indices = set(df.index)

    try:
        sq = validate_query(raw_query if isinstance(raw_query, dict) else {})
    except Exception:
        sq = StructuredQuery()

    try:
        current_id = id(df)
        if CACHE_ID != current_id or NORMALIZED_CACHE is None:
            NORMALIZED_CACHE = normalize_inputs(df.copy())
            CACHE_ID = current_id
            MASK_CACHE.clear()
        working = NORMALIZED_CACHE.copy()
    except Exception:
        working = normalize_inputs(df.copy())

    if "final_price" not in working.columns:
        log.warning("[filters] missing final_price column — proceeding without price filtering")

    try:
        working = _defensive_drop_incomplete(working)
    except Exception:
        pass

    if len(working) == 0:
        return [], _empty_like(df)

    if sq.category is None and sq.category_allowlist is None and sq.query:
        try:
            tokens = re.findall(r"[a-z0-9]+", str(sq.query).lower())
            for token in tokens:
                resolved = _CATEGORY_CANONICAL_ALIASES.get(token)
                if resolved and resolved not in GENERIC_CATEGORIES:
                    sq = sq.model_copy(update={"category": resolved})
                    break
        except Exception:
            pass

    try:
        filtered = _apply_category_stage(working, sq)
        filtered = apply_brand_filter(filtered, sq, full_df=working)
        filtered = apply_price_filter(filtered, sq)
        filtered = filtered.loc[filtered.index.intersection(valid_indices)]
    except Exception as exc:
        log.warning("[filters] pipeline error: %s — returning empty", exc)
        filtered = _empty_like(working)

    if len(filtered) > 500:
        filtered = filtered.head(500)

    filtered_indices = filtered.index.values.tolist()

    log.info(
        "filter_products complete | final candidates=%d  (%s of original)",
        len(filtered_indices), _pct(total, len(filtered_indices)),
    )

    if len(MASK_CACHE) > 100:
        MASK_CACHE.clear()

    return filtered_indices, filtered
