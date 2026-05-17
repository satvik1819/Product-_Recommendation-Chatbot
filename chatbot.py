from __future__ import annotations

import re
import traceback
from typing import Any

try:
    import importlib
    _anthropic = importlib.import_module("anthropic")
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic = None
    _HAS_ANTHROPIC = False

try:
    from src.retrieval.engine import search_with_filters
except ImportError:
    def search_with_filters(query: str, client: Any = None, top_n: int = 5) -> list:
        return []

try:
    from src.retrieval.engine import get_product_by_index as _get_product_by_index
    _HAS_GPBI = True
except ImportError:
    _HAS_GPBI = False
    def _get_product_by_index(index: int) -> dict:
        return {}

GREETING_TOKENS = {"hi", "hello", "hey", "howdy", "greetings", "yo", "sup", "hiya"}

_STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "for", "of", "and",
    "or", "to", "me", "my", "show", "find", "get", "some", "any",
}

REFINEMENT_SIGNALS = {
    "cheap", "cheapest", "best", "top", "rated", "expensive", "premium",
    "budget", "affordable", "under", "above", "over", "below", "faster",
    "lighter", "smaller", "bigger", "popular", "discount", "offer",
    "similar", "more", "another",
}

NEW_SEARCH_KEYWORDS = [
    "headset", "phone", "laptop", "watch", "shoes",
    "nike", "sony", "boat", "noise", "samsung",
]

def make_state() -> dict:
    return {
        "products": [],
        "last_query": "",
        "context_tokens": set(),
        "chat_history": [],
        "user_preferences": {},
        "category": None,
        "last_valid_products": [],
        "_last_failed_query": None,
    }


def _init_state(state: dict | None) -> dict:
    if state is None:
        state = make_state()
    state.setdefault("products", [])
    state.setdefault("last_query", "")
    state.setdefault("context_tokens", set())
    state.setdefault("chat_history", [])
    state.setdefault("user_preferences", {})
    state.setdefault("category", None)
    state.setdefault("last_valid_products", [])
    state.setdefault("_last_failed_query", None)
    return state


def normalize_query(raw: str) -> str:
    q = (raw or "").strip()
    q = re.sub(r"([a-z])([A-Z])", r"\1 \2", q)
    q = re.sub(r"([a-z])(\d)", r"\1 \2", q)
    q = re.sub(r"(\d)([a-z])", r"\1 \2", q)
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def safe_price(val: Any) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except (ValueError, TypeError):
        return 0.0


def format_price_display(val: float) -> str:
    return f"₹{val:,.0f}" if val > 0 else "N/A"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def format_products(products: list[dict]) -> list[dict]:
    out = []
    for p in products:
        if not isinstance(p, dict):
            continue
        price_raw = safe_price(
            p.get("price") or p.get("final_price") or p.get("selling_price") or 0
        )
        out.append({
            "name": str(p.get("name", "Unknown Product")),
            "price": price_raw,
            "price_display": format_price_display(price_raw),
            "rating": _safe_float(p.get("rating", p.get("ratings", 0))),
            "reviews": p.get("reviews", p.get("no_of_ratings", 0)),
            "image": str(p.get("image_url", p.get("image", ""))),
            "url": str(p.get("link", p.get("url", ""))),
            "badge": str(p.get("badge", "")),
            "category": str(p.get("category", p.get("main_category", ""))),
            "brand": str(p.get("brand", "")),
            "description": str(p.get("description", "")),
        })
    return out


def normalize_engine_output(raw: Any) -> list[dict]:
    if raw is None:
        return []
    try:
        import pandas as pd
        if isinstance(raw, pd.DataFrame):
            return raw.to_dict(orient="records")
    except Exception:
        pass
    try:
        items = list(raw)
    except Exception:
        return []
    if not items:
        return []
    results = []
    for item in items:
        try:
            if hasattr(item, "to_dict") and callable(item.to_dict):
                d = item.to_dict()
                if isinstance(d, dict):
                    results.append(d)
                continue
            if isinstance(item, dict):
                results.append(item)
                continue
            if isinstance(item, tuple):
                if _HAS_GPBI and len(item) >= 1:
                    product = _get_product_by_index(item[0])
                    if product and isinstance(product, dict):
                        results.append(product)
                continue
            if isinstance(item, str):
                results.append({"name": item})
                continue
            if hasattr(item, "__dict__"):
                data = vars(item)
                if data:
                    results.append(data)
                continue
        except Exception:
            continue
    return results


_serialise_products = format_products


def detect_intent(query: str, state: dict) -> str:
    q = query.lower().strip()
    tokens = set(q.split())
    prev = state.get("context_tokens", set()) or set()
    curr = tokens - _STOPWORDS
    has_ref = bool(tokens & REFINEMENT_SIGNALS)
    has_state = bool(state.get("products"))
    if prev:
        overlap = bool(prev & curr)
        pure_ref = has_ref and len(curr) <= 3
        topic_shift = not (overlap or pure_ref)
    else:
        topic_shift = True
    if tokens <= GREETING_TOKENS:
        return "greeting"
    if "compare" in q or " vs " in q:
        return "comparison"
    if has_state and not topic_shift and (has_ref or "similar" in q or len(tokens) <= 3):
        return "refinement"
    return "search"


def apply_followup_logic(
    query: str,
    products: list[dict],
) -> tuple[list[dict], list[str]]:
    if not products:
        return products, []
    q = query.lower()
    ops: list[str] = []
    if "similar" in q:
        ops.append("return_similar")
        return products, ops
    if any(kw in q for kw in ("cheap", "cheapest", "budget", "affordable", "low price")):
        ops.append("sorted_by_price_asc")
        return sorted(products, key=lambda x: x.get("price", 0))[:5], ops
    if any(kw in q for kw in ("expensive", "premium", "luxury", "high end")):
        ops.append("sorted_by_price_desc")
        return sorted(products, key=lambda x: x.get("price", 0), reverse=True)[:5], ops
    if any(kw in q for kw in ("best", "top", "highest rated", "most popular", "recommended")):
        ops.append("sorted_by_rating_desc")
        return sorted(products, key=lambda x: x.get("rating", 0), reverse=True)[:5], ops
    match = re.search(r"(?:under|below|less than|<)\s*[₹]?(\d[\d,]*)", q)
    if match:
        threshold = safe_price(match.group(1))
        filtered = [p for p in products if 0 < p.get("price", 0) <= threshold]
        if filtered:
            ops.append(f"filtered_under_{int(threshold)}")
            return sorted(filtered, key=lambda x: x.get("price", 0)), ops
    match = re.search(r"(?:above|over|more than|>)\s*[₹]?(\d[\d,]*)", q)
    if match:
        threshold = safe_price(match.group(1))
        filtered = [p for p in products if p.get("price", 0) >= threshold]
        if filtered:
            ops.append(f"filtered_above_{int(threshold)}")
            return sorted(filtered, key=lambda x: x.get("price", 0)), ops
    for p in products:
        brand = str(p.get("brand", "")).lower()
        if brand and brand in q:
            bf = [x for x in products if str(x.get("brand", "")).lower() == brand]
            if bf:
                ops.append(f"filtered_brand_{brand}")
                return bf, ops
    ops.append("no_op")
    return products, ops


def fetch_products(query: str, client: Any = None, top_n: int = 8) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    try:
        raw = search_with_filters(q, client, top_n=top_n)
        return format_products(normalize_engine_output(raw))
    except Exception:
        traceback.print_exc()
        return []


def extract_signals(query: str) -> list[str]:
    q = query.lower()
    signals = []
    if any(kw in q for kw in ("cheap", "budget", "affordable")):
        signals.append("price_sensitive")
    if any(kw in q for kw in ("best", "top", "rated")):
        signals.append("quality_focused")
    if re.search(r"under|below|above|over", q):
        signals.append("price_filter")
    return signals


def score_confidence(products: list[dict], intent: str) -> str:
    if intent == "greeting" or not products:
        return "fallback"
    if len(products) >= 5:
        return "strong"
    if len(products) >= 2:
        return "few"
    return "fallback"


def build_response(
    summary: str,
    products: list[dict],
    explanation: str,
    follow_up: list[str],
    signals: list[str],
    confidence: str,
    state: dict,
) -> dict:
    return {
        "summary": summary,
        "explanation": explanation,
        "products": products,
        "follow_up": follow_up,
        "signals": signals,
        "confidence": confidence,
        "result_count": len(products),
        "state": state,
    }


def safe_response(error_msg: str, state: dict) -> dict:
    return build_response(
        summary="Something went wrong. Please try again.",
        products=state.get("products", []),
        explanation=error_msg,
        follow_up=[],
        signals=[],
        confidence="fallback",
        state=state,
    )


def _update_context_tokens(state: dict, clean_query: str) -> None:
    tokens = {
        w for w in clean_query.lower().split()
        if w not in _STOPWORDS and len(w) > 2
    }
    state["context_tokens"] = tokens


def _make_anthropic_client() -> Any:
    if not _HAS_ANTHROPIC:
        return None
    try:
        return _anthropic.Anthropic()
    except Exception:
        traceback.print_exc()
        return None


def call_llm(user_message: str, client: Any, **kwargs) -> str | None:
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": user_message}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def build_explanation(
    query: str,
    products: list[dict],
    intent: str,
    client: Any,
    chat_history: list[dict] | None = None,
    user_preferences: dict | None = None,
    reviews: list | None = None,
) -> str:
    if not products or intent == "greeting":
        return ""
    top = products[0]
    default = (
        f"{top.get('name', '')} is a strong match. "
        "Here are options from the catalogue."
    )
    if client is None:
        return default
    return call_llm(user_message=query, client=client) or default


def generate_followups(products: list[dict], intent: str, category: str | None) -> list[str]:
    if not products:
        return []
    suggestions = []
    if intent in ("search", "refinement"):
        suggestions.append("Show me the cheapest options")
        suggestions.append("Which has the best rating?")
    if category:
        suggestions.append(f"More {category} under ₹5000")
    return suggestions[:3]


def handle_user_query(query: str, state: dict | None = None, client: Any = None) -> dict:
    state = _init_state(state)
    q_in = str(query or "").strip()
    clean = normalize_query(q_in)

    if not clean:
        return build_response(
            summary="Please enter a search.",
            products=[],
            explanation="",
            follow_up=["Headphones under ₹2000", "Samsung phones", "Laptops"],
            signals=[],
            confidence="fallback",
            state=state,
        )

    signals = extract_signals(clean)
    intent = detect_intent(clean, state)

    if intent == "greeting":
        reply = "Hi — what product are you looking for?"
        state["chat_history"].append({"role": "user", "content": clean})
        state["chat_history"].append({"role": "assistant", "content": reply})
        _update_context_tokens(state, clean)
        return build_response(
            summary=reply,
            products=[],
            explanation="",
            follow_up=["Wireless headphones", "Phones under ₹30000", "Perfume for men"],
            signals=signals,
            confidence="fallback",
            state=state,
        )

    if intent == "refinement" and state.get("products"):
        state["chat_history"].append({"role": "user", "content": clean})
        if any(word in clean for word in NEW_SEARCH_KEYWORDS):
            products = fetch_products(clean, client, top_n=8)
        else:
            products, _ = apply_followup_logic(clean, list(state["products"]))
            if not products:
                products = list(state["products"])
        state["products"] = products
        state["last_valid_products"] = list(products)
        summary = f"Updated results for «{clean}»."
        explanation = build_explanation(
            clean, products, intent, client,
            chat_history=state["chat_history"],
            user_preferences=state["user_preferences"],
        )
        state["chat_history"].append({"role": "assistant", "content": explanation or summary})
        _update_context_tokens(state, clean)
        conf = score_confidence(products, intent)
        return build_response(
            summary=summary,
            products=products,
            explanation=explanation,
            follow_up=generate_followups(products, intent, state.get("category")),
            signals=signals,
            confidence=conf,
            state=state,
        )

    if intent == "comparison" and len(state.get("products") or []) >= 2:
        state["chat_history"].append({"role": "user", "content": clean})
        p1, p2 = state["products"][0], state["products"][1]
        explanation = (
            f"**{p1.get('name')}** vs **{p2.get('name')}** — "
            f"compare ratings and prices in the cards."
        )
        state["chat_history"].append({"role": "assistant", "content": explanation})
        _update_context_tokens(state, clean)
        return build_response(
            summary="Comparison",
            products=[p1, p2],
            explanation=explanation,
            follow_up=generate_followups([p1, p2], intent, state.get("category")),
            signals=signals,
            confidence="few",
            state=state,
        )

    state["chat_history"].append({"role": "user", "content": clean})
    products = fetch_products(q_in, client, top_n=8)
    state["products"] = products
    state["last_valid_products"] = list(products)
    state["category"] = None
    state["last_query"] = clean
    _update_context_tokens(state, clean)

    summary = f"Results for «{clean}»." if products else "No matching products in the catalogue."
    explanation = build_explanation(
        clean, products, intent, client,
        chat_history=state["chat_history"],
        user_preferences=state["user_preferences"],
    )
    state["chat_history"].append({"role": "assistant", "content": explanation or summary})
    conf = score_confidence(products, intent)
    return build_response(
        summary=summary,
        products=products,
        explanation=explanation,
        follow_up=generate_followups(products, intent, None),
        signals=signals,
        confidence=conf,
        state=state,
    )


handle_chat = handle_user_query


def make_client() -> Any:
    return _make_anthropic_client()
