from __future__ import annotations

import logging
import math
import re
from typing import Any

log = logging.getLogger(__name__)

W_SIMILARITY = 0.22
W_RATING = 0.18
W_POPULARITY = 0.15
W_BRAND_MATCH = 0.28
W_CATEGORY_MATCH = 0.17

PENALTY_WRONG_CATEGORY = 0.85
PENALTY_WRONG_BRAND = 0.35

DIVERSITY_PENALTY_PER_REPEAT = 0.08


def _norm_rating(r: float) -> float:
    return max(0.0, min(1.0, float(r) / 5.0))


def _row_category_blob(c: dict[str, Any]) -> str:
    parts = [
        str(c.get("category", "")),
        str(c.get("main_category", "")),
        str(c.get("sub_category", "")),
    ]
    return " ".join(parts).lower()


def _category_match_score(intended: str | None, c: dict[str, Any]) -> float:
    if not intended:
        return 0.5
    ic = intended.strip().lower()
    if not ic:
        return 0.5
    hay = _row_category_blob(c)
    if re.search(rf"\b{re.escape(ic)}\b", hay):
        return 1.0
    if ic in hay:
        return 0.85
    return 0.0


def _brand_match_score(parsed_brand: str | None, c: dict[str, Any]) -> float:
    if not parsed_brand:
        return 0.5
    pb = parsed_brand.strip().lower()
    if not pb:
        return 0.5
    b = str(c.get("brand", "")).lower()
    n = str(c.get("name", "")).lower()
    if re.search(rf"\b{re.escape(pb)}\b", b):
        return 1.0
    if re.search(rf"\b{re.escape(pb)}\b", n):
        return 0.95
    if pb in b or pb in n:
        return 0.7
    return 0.0


def rank_results(
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    parsed_brand: str | None = None,
    intended_category: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        log.warning("rank_results called with empty candidate list")
        return []

    max_ratings = max(float(c.get("no_of_ratings", 0)) for c in candidates)
    if max_ratings <= 0:
        max_ratings = 1.0

    log.info(
        "Ranker | candidates=%d top_n=%d max_ratings=%.0f",
        len(candidates), top_n, max_ratings,
    )

    for c in candidates:
        similarity_score = float(c.get("similarity_score", 0.0))
        rating_score = _norm_rating(float(c.get("ratings", 0.0)))
        no_of_ratings = float(c.get("no_of_ratings", 0))
        popularity_score = math.log(1.0 + no_of_ratings) / math.log(1.0 + max_ratings)

        cat_ms = _category_match_score(intended_category, c)
        br_ms = _brand_match_score(parsed_brand, c)

        base = (
            W_SIMILARITY * similarity_score
            + W_RATING * rating_score
            + W_POPULARITY * popularity_score
            + W_BRAND_MATCH * br_ms
            + W_CATEGORY_MATCH * cat_ms
        )

        if intended_category and cat_ms < 0.5:
            base -= PENALTY_WRONG_CATEGORY
        if parsed_brand and br_ms < 0.5:
            base -= PENALTY_WRONG_BRAND

        c["final_score"] = max(0.0, base)

    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("final_score", 0.0)),
        reverse=True,
    )

    seen_brands: dict[str, int] = {}
    for item in sorted_candidates:
        brand = str(item.get("brand", "unknown")).lower().strip()
        repeat_count = seen_brands.get(brand, 0)
        if repeat_count > 0:
            item["final_score"] = max(
                0.0,
                float(item["final_score"]) - repeat_count * DIVERSITY_PENALTY_PER_REPEAT,
            )
        seen_brands[brand] = repeat_count + 1

    sorted_candidates.sort(
        key=lambda x: float(x.get("final_score", 0.0)),
        reverse=True,
    )

    result = sorted_candidates[:top_n]

    log.info(
        "Ranker | output=%d | top=%r final_score=%.4f",
        len(result),
        result[0].get("name", "")[:50] if result else "",
        float(result[0].get("final_score", 0.0)) if result else 0.0,
    )

    return result
