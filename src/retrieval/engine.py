from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from src.retrieval.filters import (
    filter_products,
    GENERIC_KEYWORDS,
    get_related_canonicals,
    resolve_query_category,
)
from src.retrieval.faiss_index import search as faiss_search, load_metadata
from src.ranking.ranker import rank_results
from src.llm.slot_extractor import extract_slots
from src.llm.validator import validate_slots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INVALID_BRAND_TOKENS: frozenset[str] = frozenset({
    "gaming", "bluetooth", "running", "laptop", "mobile",
    "phone", "speaker", "headphones", "watch", "shoes",
    "cheap", "best", "new", "latest", "all", "everything",
    "smart", "show", "hand", "cotton", "men", "women",
    "headphone", "watches", "shirt", "earphone", "earbud",
    "good", "premium", "pro", "mini", "plus", "ultra",
    "max", "lite", "air", "slim", "thin", "gel", "pens", "pen",
    "wireless", "wired", "cancelling",
})

COMMON_WORDS: frozenset[str] = frozenset({
    "smart", "show", "hand", "cotton", "men", "women",
    "for", "with", "under", "best", "new", "latest",
    "headphone", "headphones", "watch", "watches",
    "shirt", "shoes", "phone", "mobile",
    "good", "cheap", "premium", "pro", "mini", "plus", "ultra",
    "max", "lite", "air", "slim", "thin",
    "silk", "leather", "wool", "linen", "polyester", "rubber",
    "man", "woman", "kids", "boys", "girls", "unisex",
    "face", "body", "hair", "skin",
    "find", "give", "need", "want", "buy", "get",
    "below", "above", "from", "and", "the", "that", "this",
    "home", "life", "style", "store", "world", "point", "one",
    "black", "white", "blue", "red", "green", "pink", "grey", "gray",
})

CATEGORY_KEYWORD_HINTS: dict[str, str] = {
    "spice": "grocery", "spices": "grocery",
    "masala": "grocery", "biryani": "grocery", "rice": "grocery",
    "dal": "grocery", "flour": "grocery", "oil": "grocery",
    "tea": "grocery", "coffee": "grocery", "sugar": "grocery",
    "laptop": "laptop", "notebook": "laptop",
    "tablet": "tablet",
    "tv": "television", "television": "television",
    "saree": "saree", "kurta": "kurta", "jeans": "jeans",
    "legging": "legging", "dupatta": "dupatta",
    "shampoo": "hair", "moisturizer": "skin", "lipstick": "makeup",
    "serum": "skin",
    "gel pen": "pen", "ball pen": "pen", "perfume": "perfume",
}

PHONE_QUERY_HINTS = (
    "phone", "mobile", "smartphone", "iphone", "galaxy", "pixel",
    "oneplus", "oppo", "vivo", "realme", "redmi", "samsung",
)
LAPTOP_QUERY_HINTS = ("laptop", "notebook", "macbook", "chromebook", "ultrabook")
PERFUME_QUERY_HINTS = ("perfume", "fragrance", "cologne", "deo")
ACCESSORY_TERMS = (
    "charger", "charging", "adapter", "cable", "case", "cover",
    "tempered", "protector", "screen guard", "holder", "stand",
    "power bank", "earphone", "earbud", "headphone",
)
BAG_TERMS = ("backpack", "bag", "sling", "sleeve", "briefcase")
SHIRT_TERMS = ("shirt", "trouser", "jean", "kurta", "t-shirt", "tee")
INVALID_BRANDS = {"only", "not", "all", "new", "best", "latest"}
PRODUCT_KEYWORDS = {
    "headset", "headphone", "phone", "laptop",
    "watch", "shoe", "perfume",
}

INITIAL_TOP_K = 100
TOP_K_STEPS = [100, 200, 400, 500]
MAX_TOP_K = 500
MIN_RESULTS = 3
MIN_SCORE_THRESHOLD = 0.22

_df_cache: pd.DataFrame | None = None


def _get_cached_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        log.info("Cold-start: loading metadata into process cache...")
        _df_cache = load_metadata()
        log.info("Metadata cache ready: %d rows.", len(_df_cache))
    return _df_cache


def _is_valid_brand(token: str, known_brands: set) -> bool:
    if not token:
        return False
    token = token.strip().lower()
    if token in INVALID_BRAND_TOKENS:
        return False
    if len(token) < 2:
        return False
    if token not in known_brands:
        return False
    return True


def _engine_normalize_query(q: str) -> str:
    q = re.sub(r"([a-z])([A-Z])", r"\1 \2", q)
    q = re.sub(r"([a-z])(\d)", r"\1 \2", q)
    q = re.sub(r"(\d)([a-z])", r"\1 \2", q)
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


@dataclass
class ProductResult:
    faiss_index:      int
    similarity_score: float
    final_score:      float
    name:             str
    main_category:    str
    sub_category:     str
    final_price:      int
    ratings:          float
    no_of_ratings:    int
    brand:            str
    image:            str
    link:             str
    embedding_text:   str
    fallback_used:    bool = field(default=False, repr=False)

    @classmethod
    def from_dict(cls, d: dict, fallback: bool = False) -> "ProductResult":
        raw_link = d.get("link") or d.get("product_url") or d.get("url") or ""
        if raw_link and not str(raw_link).startswith("http"):
            raw_link = ""
        return cls(
            faiss_index      = int(d.get("_faiss_index", -1)),
            similarity_score = round(float(d.get("similarity_score", 0.0)), 6),
            final_score      = round(float(d.get("final_score", 0.0)), 6),
            name             = str(d.get("name", "")),
            main_category    = str(d.get("main_category", "")),
            sub_category     = str(d.get("sub_category", "")),
            final_price      = int(d.get("final_price", d.get("price", 0))),
            ratings          = float(d.get("ratings", 0.0)),
            no_of_ratings    = int(d.get("no_of_ratings", 0)),
            brand            = str(d.get("brand", "unknown")),
            image            = str(d.get("image_url", d.get("image", ""))),
            link             = str(raw_link),
            embedding_text   = str(d.get("embedding_text", ""))[:300],
            fallback_used    = fallback,
        )

    @classmethod
    def from_row(
        cls,
        row: pd.Series,
        idx: int,
        score: float,
        fallback: bool = False,
    ) -> "ProductResult":
        sim = round(float(score), 6)
        raw_link = row.get("link") or row.get("product_url") or row.get("url") or ""
        if raw_link and not str(raw_link).startswith("http"):
            raw_link = ""
        return cls(
            faiss_index      = idx,
            similarity_score = sim,
            final_score      = sim,
            name             = str(row.get("name", "")),
            main_category    = str(row.get("main_category", "")),
            sub_category     = str(row.get("sub_category", "")),
            final_price      = int(row.get("final_price", row.get("price", 0))),
            ratings          = float(row.get("ratings", 0.0)),
            no_of_ratings    = int(row.get("no_of_ratings", 0)),
            brand            = str(row.get("brand", "unknown")),
            image            = str(row.get("image_url", row.get("image", ""))),
            link             = str(raw_link),
            embedding_text   = str(row.get("embedding_text", ""))[:300],
            fallback_used    = fallback,
        )


def _validate_query_string(query_string: str) -> str:
    if not isinstance(query_string, str) or not query_string.strip():
        raise ValueError(
            f"Invalid query_string: {query_string!r}. Must be a non-empty string."
        )
    return query_string.strip()


def _meaningful_query_tokens(query_string: str) -> list[str]:
    q = _engine_normalize_query(query_string)
    out: list[str] = []
    for raw in q.split():
        t = raw.strip().lower()
        if len(t) <= 2:
            continue
        if t.isdigit():
            continue
        if t in GENERIC_KEYWORDS:
            continue
        out.append(t)
    return out


def _prepare_faiss_query(query_string: str) -> tuple[str, bool]:
    normalized = _engine_normalize_query(query_string)
    tokens = _meaningful_query_tokens(query_string)
    if not tokens:
        return (normalized or query_string.strip(), True)
    joined = " ".join(tokens).strip()
    if not joined:
        return (normalized or query_string.strip(), True)
    return (joined, False)


def _word_token_match_any(name: str, tokens: list[str]) -> bool:
    name_lower = (name or "").lower()
    for tok in tokens:
        if re.search(rf"\b{re.escape(tok)}\b", name_lower):
            return True
    return False


def _row_matches_intended_category(
    df: pd.DataFrame,
    idx: int,
    intended_category: str | None,
    soft_related: frozenset[str] | None = None,
) -> bool:
    if not intended_category:
        return True
    cat = str(intended_category).strip().lower()
    if not cat:
        return True
    try:
        row = df.iloc[idx]
    except (IndexError, AttributeError):
        return False
    toks = row.get("canonical_tokens")
    if toks is not None and cat in toks:
        return True
    hay = " ".join([
        str(row.get("category", "")),
        str(row.get("main_category", "")),
        str(row.get("sub_category", "")),
    ]).lower()
    if re.search(rf"\b{re.escape(cat)}\b", hay):
        return True
    if soft_related and toks is not None:
        if toks & soft_related:
            return True
    return False


def _is_relevant_pair(
    idx: int,
    score: float,
    df: pd.DataFrame,
    query_string: str,
    parsed_brand: str | None,
    high_similarity_threshold: float = 0.62,
) -> bool:
    try:
        row = df.iloc[idx]
    except (IndexError, AttributeError):
        return False
    name = str(row.get("name", "")).lower()
    brand = str(row.get("brand", "")).lower()
    strong_tokens = _meaningful_query_tokens(query_string)
    token_match = _word_token_match_any(name, strong_tokens) if strong_tokens else False
    brand_norm = (parsed_brand or "").strip().lower()
    brand_match = False
    if brand_norm:
        brand_match = bool(re.search(rf"\b{re.escape(brand_norm)}\b", brand)) or bool(
            re.search(rf"\b{re.escape(brand_norm)}\b", name)
        )
    high_similarity = float(score) >= high_similarity_threshold
    return token_match or brand_match or high_similarity


def _run_faiss(query_string: str, top_k: int, df: pd.DataFrame) -> list[tuple[int, float]]:
    safe_k = min(top_k, len(df))
    results = faiss_search(query_string, top_k=safe_k)
    pairs: list[tuple[int, float]] = []
    for r in results:
        idx = r.get("_faiss_index", -1)
        score = r.get("similarity_score", 0.0)
        if idx >= 0:
            pairs.append((int(idx), float(score)))
    return pairs


def _intersect(
    faiss_pairs: list[tuple[int, float]],
    candidate_set: set[int],
    min_score: float,
) -> list[tuple[int, float]]:
    seen: set[int] = set()
    result: list[tuple[int, float]] = []
    for idx, score in faiss_pairs:
        if idx in seen:
            continue
        seen.add(idx)
        if idx in candidate_set and score >= min_score:
            result.append((idx, score))
    return result


def _faiss_semantic_cleanup(
    faiss_pairs: list[tuple[int, float]],
    df: pd.DataFrame,
    query_string: str,
    parsed_brand: str | None,
    candidate_set: set[int] | None,
) -> list[tuple[int, float]]:
    filtered: list[tuple[int, float]] = []
    for idx, score in faiss_pairs:
        if candidate_set is not None and idx not in candidate_set:
            continue
        if _is_relevant_pair(idx, score, df, query_string, parsed_brand):
            filtered.append((idx, score))
    return filtered


def _metadata_from_candidate_set(
    candidate_set: set[int],
    df: pd.DataFrame,
    cap: int,
) -> list[tuple[int, float]]:
    if not candidate_set:
        return []
    try:
        sub = df.loc[sorted(candidate_set)]
    except Exception:
        return []
    sub = sub.sort_values(
        ["no_of_ratings", "ratings"],
        ascending=[False, False],
        kind="mergesort",
    ).head(cap)
    return [(int(i), 0.0) for i in sub.index]


def _product_type_block(query_norm: str, name: str, main_cat: str, sub_cat: str) -> bool:
    q = query_norm.lower()
    n = (name or "").lower()
    blob = f"{main_cat} {sub_cat}".lower()

    phone_q = any(h in q for h in PHONE_QUERY_HINTS) or re.search(
        r"\bs\d{1,2}\b", q
    ) is not None
    if phone_q:
        block_terms = (
            "charger", "charging", "adapter", "case", "cover",
            "protector", "tempered", "power bank",
        )
        if not any(t in q for t in block_terms):
            if any(t in n for t in block_terms):
                return True

    lap_q = any(h in q for h in LAPTOP_QUERY_HINTS)
    if lap_q:
        if any(b in n for b in BAG_TERMS) and not any(b in q for b in BAG_TERMS):
            return True

    if any(h in q for h in PERFUME_QUERY_HINTS):
        if any(s in n for s in SHIRT_TERMS) and not any(s in q for s in SHIRT_TERMS):
            return True

    return False


def _relax_metadata_candidates(
    df: pd.DataFrame,
    parsed_query: dict,
) -> tuple[set[int], pd.DataFrame, str, dict]:
    base = {
        "query": parsed_query.get("query"),
        "category": parsed_query.get("category"),
        "brand": parsed_query.get("brand"),
        "min_price": parsed_query.get("min_price"),
        "max_price": parsed_query.get("max_price"),
        "keywords": parsed_query.get("keywords") or [],
    }
    orig_cat = base.get("category")

    idx, fdf = filter_products(df, base)
    if len(idx):
        log.info("pipeline | relaxation_step=strict | candidates=%d", len(idx))
        return set(idx), fdf, "strict", base

    r1 = {**base, "min_price": None, "max_price": None}
    idx, fdf = filter_products(df, r1)
    if len(idx):
        log.info("pipeline | relaxation_step=remove_price | candidates=%d", len(idx))
        return set(idx), fdf, "remove_price", r1

    r2 = {**r1, "brand": None}
    idx, fdf = filter_products(df, r2)
    if len(idx):
        log.info("pipeline | relaxation_step=relax_brand | candidates=%d", len(idx))
        return set(idx), fdf, "relax_brand", r2

    if orig_cat:
        related = get_related_canonicals(orig_cat)
        r3 = {**r2, "category": None, "category_allowlist": related}
        idx, fdf = filter_products(df, r3)
        if len(idx):
            log.info(
                "pipeline | relaxation_step=related_category | candidates=%d | group=%s",
                len(idx), related,
            )
            return set(idx), fdf, "related_category", r3

    log.warning("pipeline | relaxation_step=exhausted | candidates=0")
    return set(), fdf, "failed", r2


def _final_validate_results(
    results: list[ProductResult],
    df: pd.DataFrame,
    query_string: str,
    parsed_brand: str | None,
    intended_canon: str | None,
    strict_category: bool,
    soft_related: frozenset[str] | None,
    relax_step: str,
) -> list[ProductResult]:
    qn = _engine_normalize_query(query_string)
    out: list[ProductResult] = []
    for r in results:
        try:
            row = df.iloc[r.faiss_index]
        except Exception:
            continue
        name = str(row.get("name", ""))
        mc = str(row.get("main_category", ""))
        sc = str(row.get("sub_category", ""))
        if _product_type_block(qn, name, mc, sc):
            continue
        if intended_canon:
            if strict_category:
                if not _row_matches_intended_category(
                    df, r.faiss_index, intended_canon, None
                ):
                    continue
            elif relax_step == "related_category" and soft_related:
                if not _row_matches_intended_category(
                    df, r.faiss_index, intended_canon, soft_related
                ):
                    continue
        if not _is_relevant_pair(
            r.faiss_index, r.similarity_score, df, query_string, parsed_brand, 0.45
        ):
            continue
        out.append(r)
    return out


def get_product_by_index(index: int) -> dict:
    df = _get_cached_df()
    try:
        if index < 0 or index >= len(df):
            return {}
        return df.iloc[int(index)].to_dict()
    except Exception:
        return {}


def search_with_filters(
    query_string: str,
    groq_client,
    top_n: int = 5,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> list[ProductResult]:
    log.info("─" * 60)
    try:
        query_string = _validate_query_string(query_string)
    except ValueError as exc:
        log.error("search_with_filters: %s", exc)
        return []

    query_string = re.sub(r"([a-z])([A-Z])", r"\1 \2", query_string)
    query_string = re.sub(r"([a-z])(\d)", r"\1 \2", query_string)
    query_string = re.sub(r"(\d)([a-z])", r"\1 \2", query_string)
    query_string = re.sub(r"\s+", " ", query_string).strip()

    _FALLBACK_PARSED: dict = {
        "query": query_string,
        "max_price": None,
        "min_price": None,
        "category": None,
        "brand": None,
        "keywords": [],
    }

    try:
        raw_slots = extract_slots(query_string, groq_client)
        validated = validate_slots(raw_slots)
        parsed_query = {
            "query": query_string,
            "category": validated.get("category"),
            "min_price": validated.get("price", {}).get("min"),
            "max_price": validated.get("price", {}).get("max"),
            "brand": (
                validated.get("brand", {}).get("include")[0]
                if validated.get("brand", {}).get("include")
                else None
            ),
            "keywords": validated.get("features", {}).get("required", []),
        }
    except Exception as exc:
        log.error("Parser failed unexpectedly: %s", exc, exc_info=True)
        parsed_query = _FALLBACK_PARSED

    if not parsed_query.get("query"):
        parsed_query = {**parsed_query, "query": query_string}

    df = _get_cached_df()
    if not hasattr(df, "_known_brands_cache"):
        df._known_brands_cache = set(
            df["brand"].dropna().astype(str).str.lower().str.strip().unique()
        )

    for kw in list(parsed_query.get("keywords") or []):
        kw_clean = kw.strip().lower()
        if kw_clean in COMMON_WORDS or len(kw_clean) < 2:
            continue
        if _is_valid_brand(kw_clean, df._known_brands_cache):
            log.info("Brand recovery: promoted keyword %r to brand.", kw_clean)
            parsed_query["brand"] = kw_clean
            try:
                parsed_query["keywords"].remove(kw)
            except ValueError:
                pass
            break

    if not parsed_query.get("brand"):
        known_brands = df._known_brands_cache
        query_lower = query_string.lower()
        detected_brand: str | None = None
        tokens = query_lower.split()
        for word in tokens:
            if word in COMMON_WORDS:
                continue
            if _is_valid_brand(word, known_brands) and len(word) >= 2:
                detected_brand = word
                break
        if not detected_brand:
            padded_query = f" {query_lower} "
            for brand in sorted(known_brands, key=len, reverse=True):
                if len(brand) < 3:
                    continue
                if brand in COMMON_WORDS:
                    continue
                if _is_valid_brand(brand, known_brands) and f" {brand} " in padded_query:
                    detected_brand = brand
                    break
        parsed_query["brand"] = detected_brand

    if not parsed_query.get("category"):
        ql = query_string.lower()
        for kw, cat_hint in CATEGORY_KEYWORD_HINTS.items():
            if kw in ql:
                parsed_query["category"] = cat_hint
                break

    if parsed_query["brand"]:
        b = str(parsed_query["brand"]).lower()
        if b in INVALID_BRANDS and any(k in str(parsed_query["query"]).lower() for k in PRODUCT_KEYWORDS):
            log.warning("Dropping invalid brand: %s", b)
            parsed_query["brand"] = None

    has_constraints = bool(
        parsed_query.get("brand")
        or parsed_query.get("category")
        or parsed_query.get("min_price")
        or parsed_query.get("max_price")
    )

    intended_canon = resolve_query_category(parsed_query.get("category") or "") or None
    parsed_brand = (parsed_query.get("brand") or "").strip().lower() or None
    soft_related: frozenset[str] | None = None
    if intended_canon:
        soft_related = frozenset(get_related_canonicals(intended_canon))

    faiss_query, faiss_weak = _prepare_faiss_query(query_string)
    faiss_pairs: list[tuple[int, float]] = []

    if not has_constraints:
        log.info("pipeline | mode=unconstrained_faiss | has_constraints=False")
        candidate_set = set(df.index.tolist())
        filtered_df = df
        effective_query = {
            "query": query_string,
            "category": None,
            "brand": None,
            "min_price": None,
            "max_price": None,
            "keywords": parsed_query.get("keywords") or [],
        }
        relax_step = "unconstrained"
    else:
        candidate_set, filtered_df, relax_step, effective_query = _relax_metadata_candidates(
            df, parsed_query
        )
        if not candidate_set:
            log.warning("pipeline | no_metadata_candidates_after_relaxation")
            return []

    fallback_used = False

    if not faiss_weak:
        for top_k in TOP_K_STEPS:
            faiss_pairs = _run_faiss(faiss_query, top_k, df)
            inter = _intersect(faiss_pairs, candidate_set, min_score)
            log.info(
                "pipeline | faiss_ladder | top_k=%d | inter=%d",
                top_k, len(inter),
            )
            if len(inter) >= MIN_RESULTS:
                break
            if top_k >= MAX_TOP_K:
                break
        intersection_work = _intersect(faiss_pairs, candidate_set, min_score)
        if len(intersection_work) < MIN_RESULTS:
            intersection_work = _intersect(faiss_pairs, candidate_set, 0.0)
            if intersection_work:
                fallback_used = True
                log.info("pipeline | faiss_threshold_relaxed_within_candidates")
    else:
        intersection_work = []
        log.warning("pipeline | faiss_weak_query")

    if len(intersection_work) < MIN_RESULTS and not faiss_weak:
        cleaned = _faiss_semantic_cleanup(
            faiss_pairs, df, query_string, parsed_brand, candidate_set
        )
        if len(cleaned) >= MIN_RESULTS:
            intersection_work = cleaned
            fallback_used = True
            log.info("pipeline | faiss_semantic_recovery")

    if len(intersection_work) < MIN_RESULTS:
        meta_pairs = _metadata_from_candidate_set(
            candidate_set, df, max(top_n * 5, MIN_RESULTS * 3)
        )
        meta_pairs = [
            (i, s)
            for i, s in meta_pairs
            if _is_relevant_pair(i, s, df, query_string, parsed_brand, 0.35)
        ]
        if len(meta_pairs) >= MIN_RESULTS:
            intersection_work = meta_pairs
            fallback_used = True
            log.info("pipeline | metadata_order_within_candidates")

    if not intersection_work:
        if not has_constraints and faiss_pairs:
            intersection_work = [
                (i, s) for i, s in faiss_pairs
                if _is_relevant_pair(i, s, df, query_string, parsed_brand, 0.4)
            ][: max(top_n * 4, 40)]
            fallback_used = True
            log.info("pipeline | faiss_only_unconstrained_cleanup")
        else:
            log.warning("pipeline | empty_intersection")
            return []

    strict_cat = bool(
        has_constraints
        and intended_canon
        and relax_step in ("strict", "remove_price", "relax_brand")
    )

    candidates: list[dict] = []
    seen_indices: set[int] = set()
    qn = _engine_normalize_query(query_string)

    for idx, score in intersection_work:
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        try:
            row = df.iloc[idx]
        except IndexError:
            continue
        name = str(row.get("name", ""))
        if _product_type_block(
            qn, name,
            str(row.get("main_category", "")),
            str(row.get("sub_category", "")),
        ):
            continue
        candidates.append({
            "_faiss_index": idx,
            "similarity_score": max(0.0, min(1.0, float(score))),
            "name": name,
            "category": str(row.get(
                "category",
                row.get("main_category", row.get("sub_category", "")),
            )),
            "main_category": str(row.get("main_category", "")),
            "sub_category": str(row.get("sub_category", "")),
            "price": row.get("price", row.get("final_price", 0)),
            "final_price": int(row.get("final_price", 0)),
            "ratings": float(row.get("ratings", 0.0)),
            "no_of_ratings": int(row.get("no_of_ratings", 0)),
            "brand": str(row.get("brand", "unknown")),
            "image_url": str(row.get("image_url", row.get("image", ""))),
            "image": str(row.get("image", "")),
            "link": row.get("link") or row.get("product_url") or row.get("url") or "",
            "embedding_text": str(row.get("embedding_text", ""))[:300],
        })

    for c in candidates:
        if c.get("link") and not str(c["link"]).startswith("http"):
            c["link"] = ""

    if intended_canon:
        candidates = [
            c for c in candidates
            if intended_canon.lower() in (
                str(c.get("category", "")) + " " + str(c.get("name", ""))
            ).lower()
        ]

    if not candidates:
        log.warning("pipeline | no_candidates_post_type_filter")
        return []

    try:
        ranked_dicts = rank_results(
            candidates,
            top_n=top_n,
            parsed_brand=parsed_brand,
            intended_category=intended_canon,
        )
    except Exception as exc:
        log.error("Ranking failed: %s", exc, exc_info=True)
        ranked_dicts = candidates[:top_n]

    results = [ProductResult.from_dict(d, fallback=fallback_used) for d in ranked_dicts]

    results = _final_validate_results(
        results,
        df,
        query_string,
        parsed_brand,
        intended_canon,
        strict_cat,
        soft_related,
        relax_step,
    )[:top_n]

    if results:
        log.info(
            "SUMMARY | query=%r | relax=%s | faiss_n=%d | ranked=%d",
            query_string, relax_step, len(faiss_pairs), len(results),
        )
    log.info("─" * 60)
    return results


try:
    log.info("Preloading metadata cache at import time...")
    _preload_df = _get_cached_df()
    if len(_preload_df) > 0:
        log.info("Warm-up FAISS search...")
        _ = faiss_search("test", top_k=1)
        log.info("Warm-up complete — engine ready.")
except Exception as _preload_exc:
    log.warning("Warm-up failed (non-fatal): %s", _preload_exc)


if __name__ == "__main__":
    import os
    import sys
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[ERROR] Set the GROQ_API_KEY environment variable.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    SMOKE_TESTS = [
        "gaming headphones under 2000",
        "nike running shoes",
        "cheap bluetooth speaker",
        "laptop below 50000",
        "blankets",
        "phones",
        "running shoes",
        "asdfghjkl",
    ]

    print("\n=== Smoke Tests ===")
    for qstr in SMOKE_TESTS:
        print("\n" + "=" * 65)
        print(f"TEST: {qstr!r}")
        print("=" * 65)
        try:
            results = search_with_filters(qstr, groq_client, top_n=5)
        except Exception as exc:
            print(f"[EXCEPTION] {exc}")
            continue
        if not results:
            print("  No results returned.")
            continue
        for r in results:
            fb = "[fallback]" if r.fallback_used else ""
            print(
                f"  #{r.faiss_index:<6} sim={r.similarity_score:.4f} "
                f"final={r.final_score:.4f} {fb} | "
                f"{r.name[:50]:<50} | ₹{r.final_price}"
            )
    print("\nSmoke tests complete.")
