"""
Production-grade Gradio UI — Deterministic Hybrid Recommendation Chatbot
========================================================================
Single-file. No local imports beyond chatbot + gradio. Immediately runnable.

Visual Style: Nisa AI — Obsidian Dark, Glassmorphism, Deep Indigo Glow
Typography: Playfair Display (headings) + DM Sans (body) via Google Fonts

Contract (enforced throughout):
  submit_query(query, state, chat_history) → 7 outputs (STRICT)
  Output order: chat_history, html_output, btn1, btn2, btn3, state, textbox_value

Architecture:
  Backend = Brain  |  UI = Presentation ONLY
  UI never modifies queries, re-ranks results, or infers missing data.

Alignment notes (chatbot.py ↔ app.py):
  * handle_user_query / make_state imported directly from chatbot.py
  * _INITIAL_STATE delegates to make_state() — single source of truth
  * adapt_response() removed — chatbot._serialise_products() already emits
    the exact keys _render_card expects (name/price/rating/reviews/image/url/badge)
  * signals, confidence, result_count passed through unmodified from backend
  * submit_query calls handle_user_query positionally (matches chatbot signature)
  * State fallback uses make_state() so all keys are always present
"""

import os
import re
import html
import time
from typing import Any

import gradio as gr

# ── Backend ───────────────────────────────────────────────────────────────────
# chatbot.py is the single source of truth for all business logic.
# app.py is presentation only — it NEVER re-ranks, re-maps, or re-computes.
from chatbot import handle_user_query, make_state


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GROQ CLIENT INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

try:
    from groq import Groq as _Groq
    _groq_client: Any = _Groq(api_key=os.environ["GROQ_API_KEY"]) if os.getenv("GROQ_API_KEY") else None
except Exception:
    _groq_client = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SECURITY & SANITISATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_url(url: Any) -> str | None:
    """Allow only http/https URLs with no shell-injection characters."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not re.match(r"^https?://[^\s<>\"'`\\]+$", url):
        return None
    return url


def _esc(value: Any, fallback: str = "") -> str:
    """HTML-escape a value; return fallback if None/empty."""
    if value is None or str(value).strip() == "":
        return fallback
    return html.escape(str(value))


def _fmt_price(price: Any, currency: str = "₹") -> str:
    try:
        return f"{currency}{int(float(price)):,}"
    except (TypeError, ValueError):
        return "Not available"


def _fmt_rating(rating: Any) -> tuple[str, str]:
    """Returns (stars_str, numeric_str)."""
    try:
        r      = float(rating)
        filled = round(r)
        stars  = "★" * filled + "☆" * (5 - filled)
        return stars, f"{r:.1f}"
    except (TypeError, ValueError):
        return "☆☆☆☆☆", "N/A"


def _fmt_reviews(reviews: Any) -> str:
    try:
        return f"{int(reviews):,} reviews"
    except (TypeError, ValueError):
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HTML RENDERERS
# Product cards ONLY — LLM text (summary, explanation) is NEVER rendered as HTML.
# ══════════════════════════════════════════════════════════════════════════════

def _render_card(product: dict, index: int) -> str:
    """
    Build one sanitised product card with Nisa AI glassmorphism aesthetic.
    Expects keys: name, price, rating, reviews, image, url, badge.
    These are guaranteed by chatbot._serialise_products() — no re-mapping needed.
    """
    if not isinstance(product, dict):
        return ""

    name      = _esc(product.get("name"),    "Unknown Product")
    price     = _fmt_price(product.get("price"))
    stars, rn = _fmt_rating(product.get("rating"))
    reviews   = _fmt_reviews(product.get("reviews"))
    badge     = _esc(product.get("badge"),   "")
    img_url   = _sanitize_url(product.get("image"))
    prd_url   = _sanitize_url(product.get("url"))

    if img_url:
        img_html = (
            f'<img src="{img_url}" alt="{name}" class="card-img" '
            f'onerror="this.style.display=\'none\';'
            f'this.nextSibling.style.display=\'flex\'">'
            f'<div class="card-img-fallback" style="display:none">'
            f'<svg width="36" height="36" viewBox="0 0 24 24" fill="none" '
            f'stroke="rgba(139,92,246,0.6)" stroke-width="1.5">'
            f'<rect x="3" y="3" width="18" height="18" rx="3"/>'
            f'<circle cx="8.5" cy="8.5" r="1.5"/>'
            f'<path d="M21 15l-5-5L5 21"/></svg></div>'
        )
    else:
        img_html = (
            '<div class="card-img-fallback">'
            '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" '
            'stroke="rgba(139,92,246,0.6)" stroke-width="1.5">'
            '<rect x="3" y="3" width="18" height="18" rx="3"/>'
            '<circle cx="8.5" cy="8.5" r="1.5"/>'
            '<path d="M21 15l-5-5L5 21"/></svg></div>'
        )

    badge_html = f'<span class="card-badge">{badge}</span>' if badge else ""

    link_html = (
        f'<a href="{prd_url}" target="_blank" rel="noopener noreferrer" '
        f'class="card-link">View Product ↗</a>'
        if prd_url else
        '<span class="card-link card-link-disabled">Link unavailable</span>'
    )

    return f"""
<div class="product-card" style="animation-delay:{index * 0.05:.2f}s">
  <div class="card-img-wrap">
    {img_html}
    <span class="index-badge">#{index + 1}</span>
    {badge_html}
  </div>
  <div class="card-body">
    <p class="card-name" title="{name}">{name}</p>
    <p class="card-rating">{stars} <span class="rating-num">{rn}</span></p>
    <p class="card-reviews">{reviews}</p>
    <p class="card-price">{price}</p>
  </div>
  <div class="card-footer">{link_html}</div>
</div>"""


def _render_signals(signals: list) -> str:
    if not signals:
        return ""
    tags = "".join(f'<span class="signal-tag">{_esc(s)}</span>' for s in signals)
    return (
        f'<div class="signal-bar">'
        f'<span class="signal-label">Active signals</span>{tags}</div>'
    )


def build_product_html(response: dict) -> str:
    """
    Full right-panel HTML from validated backend response.
    This is the ONLY point where dynamic content enters HTML.

    Reads directly from chatbot.py response keys:
      products     — already serialised by _serialise_products()
      signals      — list[str] built by handle_user_query()
      result_count — int set by handle_user_query()
      confidence   — "strong" | "few" | "fallback" from handle_user_query()
    """
    products   = response.get("products")    or []
    signals    = response.get("signals")     or []
    count      = response.get("result_count", len(products))
    confidence = response.get("confidence",  "fallback")

    conf_label = {
        "strong":   "Top Recommendations",
        "few":      "Limited Results",
        "fallback": "Closest Matches",
    }.get(confidence, "Results")

    conf_icon = {
        "strong":   "✦",
        "few":      "◈",
        "fallback": "◇",
    }.get(confidence, "◈")

    fallback_notice = (
        '<span class="fallback-notice">⚠ No exact match — showing closest</span>'
        if confidence == "fallback" else ""
    )

    header = (
        f'<div class="results-header">'
        f'<span class="results-title">{conf_icon} {_esc(conf_label)}</span>'
        f'<span class="results-count">{count} result{"s" if count != 1 else ""}</span>'
        f'{fallback_notice}</div>'
    )

    signals_html = _render_signals(signals)

    if not products:
        return (
            header + signals_html +
            '<div class="empty-state">'
            '<div class="empty-icon">◎</div>'
            '<p class="empty-title">No products found</p>'
            '<p class="empty-sub">Try a different search or use the suggestions below.</p>'
            '</div>'
        )

    cards = "".join(
        _render_card(p, i)
        for i, p in enumerate(products)
        if isinstance(p, dict)
    )
    if not cards:
        return header + signals_html + '<p class="error-msg">Product data is malformed.</p>'

    return f'{header}{signals_html}<div class="product-grid">{cards}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORE SUBMIT HANDLER
#
# INPUT CONTRACT:
#   submit_query(query: Any, state: Any, chat_history: Any)
#
# OUTPUT CONTRACT (7 values, strict order):
#   1. chat_history  — list of [user_str, bot_str] pairs
#   2. html_output   — str
#   3. btn1_update   — gr.update
#   4. btn2_update   — gr.update
#   5. btn3_update   — gr.update
#   6. state         — dict
#   7. textbox_value — str (always "")
# ══════════════════════════════════════════════════════════════════════════════

def _safe_return(
    chat_history: list,
    html_output: str,
    btn_updates: list,
    new_state: dict,
    textbox: str = "",
) -> tuple:
    """Always emit exactly 7 correctly-typed outputs."""
    html_output = html_output or "<div class='error-msg'>No results found.</div>"
    # Pad btn_updates to exactly 3
    while len(btn_updates) < 3:
        btn_updates.append(gr.update(visible=False))
    return (
        chat_history,
        html_output,
        btn_updates[0],
        btn_updates[1],
        btn_updates[2],
        new_state,
        textbox,
    )


def submit_query(query: Any, state: Any, chat_history: Any) -> tuple:
    """
    Main Gradio event handler.

    Follows input/output contract exactly.
    Never modifies query before passing to backend.
    All state comes exclusively from backend response.

    Alignment with chatbot.py:
      * State initialised via make_state() when missing — guarantees all keys present
      * handle_user_query called positionally — matches (query, state, client) signature
      * Response used verbatim — no field re-mapping or re-computation
      * signals, confidence, result_count forwarded as-is from backend
    """
    # ── Safe initialisation ───────────────────────────────────────────────────
    query        = str(query or "").strip()
    chat_history = list(chat_history or [])

    # Use make_state() as fallback to guarantee all chatbot.py keys are present.
    if isinstance(state, dict) and state:
        new_state = dict(state)
        for k, v in make_state().items():
            new_state.setdefault(k, v)
    else:
        new_state = make_state()

    btn_updates = [gr.update(visible=False)] * 3
    html_output = ""

    # ── Guard: empty input ────────────────────────────────────────────────────
    if not query:
        return _safe_return(chat_history, html_output, btn_updates, new_state)

    # ── Race-condition stamp ──────────────────────────────────────────────────
    request_id = time.time_ns()
    new_state["request_id"] = request_id

    # ── Append user turn ──────────────────────────────────────────────────────
    chat_history = chat_history + [[query, None]]

    try:
        # ── Backend call — positional args match chatbot.handle_user_query sig ─
        raw_response = handle_user_query(query, new_state, _groq_client)

        if not isinstance(raw_response, dict):
            raw_response = {}

        # ── Diagnostic log ────────────────────────────────────────────────────
        print("QUERY       :", query)
        print("RESULT COUNT:", raw_response.get("result_count", 0))
        print("CONFIDENCE  :", raw_response.get("confidence", "—"))
        print("SIGNALS     :", raw_response.get("signals", []))

        # ── Race-condition check ──────────────────────────────────────────────
        if new_state.get("request_id") != request_id:
            return _safe_return(chat_history, html_output, btn_updates, new_state)

        # ── State: overwrite from backend only if valid ───────────────────────
        backend_state = raw_response.get("state")
        if isinstance(backend_state, dict) and backend_state:
            new_state = backend_state
        new_state["request_id"] = request_id

        # ── Bot message: plain Markdown ONLY, never raw HTML ──────────────────
        summary     = str(raw_response.get("summary",     "") or "")
        explanation = str(raw_response.get("explanation", "") or "")
        bot_msg     = f"{summary}\n\n{explanation}".strip() or "No response received."
        chat_history[-1][1] = bot_msg

        # ── Build product panel HTML ──────────────────────────────────────────
        html_output = build_product_html(raw_response)

        # ── Follow-up button updates ──────────────────────────────────────────
        follow_up = raw_response.get("follow_up") or []
        if isinstance(follow_up, str):
            follow_up = [follow_up]
        if not isinstance(follow_up, list):
            follow_up = []

        btn_updates = [gr.update(visible=False)] * 3
        for i in range(min(3, len(follow_up))):
            label = str(follow_up[i] or "").strip()
            if label:
                btn_updates[i] = gr.update(value=label, visible=True)

        return _safe_return(chat_history, html_output, btn_updates, new_state)

    except Exception as exc:
        err_chat = chat_history[:]
        err_chat[-1][1] = (
            f"Something went wrong. Please try again.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
        return _safe_return(
            err_chat,
            "<div class='error-msg'>An error occurred. Please retry.</div>",
            [gr.update(visible=False)] * 3,
            new_state,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CSS (Nisa AI Dark Obsidian + Glassmorphism)
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Root Theme Variables ─────────────────────────────────────────────────── */
:root {
  --bg-void:        #050505;
  --bg-surface:     #0c0c14;
  --bg-raised:      #12121f;
  --bg-glass:       rgba(255,255,255,0.04);
  --bg-glass-hover: rgba(255,255,255,0.07);

  --border-subtle:  rgba(255,255,255,0.07);
  --border-glow:    rgba(99,82,246,0.35);
  --border-active:  rgba(139,92,246,0.6);

  --indigo-deep:    #1e1b4b;
  --indigo-mid:     #4338ca;
  --indigo-bright:  #818cf8;
  --indigo-glow:    rgba(99,82,246,0.15);

  --violet-accent:  #8b5cf6;
  --violet-soft:    rgba(139,92,246,0.2);

  --gold-accent:    #f6c90e;
  --amber-warm:     #f59e0b;

  --text-primary:   #f1f0ff;
  --text-secondary: #a09cc0;
  --text-muted:     #5c587a;
  --text-ghost:     #3a3758;

  --radius-card:    15px;
  --radius-pill:    999px;
  --radius-btn:     10px;

  --shadow-card:    0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px var(--border-subtle);
  --shadow-glow:    0 0 40px rgba(99,82,246,0.12);
  --transition:     0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ── Container ────────────────────────────────────────────────────────────── */
.gradio-container {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: var(--bg-void) !important;
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(67,56,202,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(139,92,246,0.08) 0%, transparent 55%) !important;
}

/* ── App Header ───────────────────────────────────────────────────────────── */
#app-header {
  text-align: center;
  padding: 32px 20px 20px;
  position: relative;
}
#app-header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 50%;
  transform: translateX(-50%);
  width: 200px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
}
.header-eyebrow {
  font-family: 'DM Sans', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--indigo-bright);
  margin-bottom: 10px;
  opacity: 0.8;
}
.header-title {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text-primary) !important;
  line-height: 1.2;
  letter-spacing: -0.02em;
  margin-bottom: 10px;
}
.header-title em {
  font-style: italic;
  background: linear-gradient(135deg, var(--indigo-bright), var(--violet-accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.header-sub {
  font-size: 13px;
  color: var(--text-muted);
  font-weight: 400;
  letter-spacing: 0.01em;
}

/* ── Panel Layout ─────────────────────────────────────────────────────────── */
#left-panel {
  display: flex;
  flex-direction: column;
  height: 620px;
  overflow: hidden;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card), var(--shadow-glow);
}

#right-panel {
  display: flex;
  flex-direction: column;
  height: 620px;
  overflow: hidden;
}

/* ── Chatbot ──────────────────────────────────────────────────────────────── */
#chat-panel {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  padding: 4px 0;
}

/* Chat panel label */
#chat-panel > .label-wrap,
#left-panel .label-wrap {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  padding: 14px 16px 10px !important;
  border-bottom: 1px solid var(--border-subtle) !important;
}

/* Gradio chatbot wrapper */
.gradio-chatbot {
  background: transparent !important;
  border: none !important;
}

/* User bubble */
.gradio-chatbot .message.user {
  background: linear-gradient(135deg, #312e81, #4338ca) !important;
  border: 1px solid rgba(99,82,246,0.3) !important;
  border-radius: 14px 14px 4px 14px !important;
  color: #e0e7ff !important;
  font-size: 13.5px !important;
  line-height: 1.55 !important;
  padding: 10px 14px !important;
  box-shadow: 0 4px 16px rgba(67,56,202,0.25) !important;
}

/* Bot bubble */
.gradio-chatbot .message.bot {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 4px 14px 14px 14px !important;
  color: var(--text-secondary) !important;
  font-size: 13.5px !important;
  line-height: 1.6 !important;
  padding: 10px 14px !important;
}

/* ── Follow-up Pill Buttons ───────────────────────────────────────────────── */
.followup-btn button, button.followup-btn {
  background: rgba(67,56,202,0.12) !important;
  border: 1px solid rgba(99,82,246,0.25) !important;
  color: var(--indigo-bright) !important;
  border-radius: var(--radius-pill) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 6px 16px !important;
  cursor: pointer !important;
  transition: all var(--transition) !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 220px !important;
  letter-spacing: 0.01em !important;
}
.followup-btn button:hover {
  background: rgba(99,82,246,0.2) !important;
  border-color: rgba(139,92,246,0.5) !important;
  color: #c4b5fd !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(99,82,246,0.2) !important;
}

/* ── Query Input ──────────────────────────────────────────────────────────── */
#query-input textarea {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-btn) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  resize: none !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
}
#query-input textarea::placeholder {
  color: var(--text-ghost) !important;
  font-style: italic;
}
#query-input textarea:focus {
  border-color: var(--border-active) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.12), 0 0 20px rgba(139,92,246,0.08) !important;
  outline: none !important;
}

/* ── Send Button ──────────────────────────────────────────────────────────── */
#send-btn {
  background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
  border: 1px solid rgba(139,92,246,0.4) !important;
  color: #e0e7ff !important;
  border-radius: var(--radius-btn) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  transition: all var(--transition) !important;
  box-shadow: 0 4px 16px rgba(67,56,202,0.3) !important;
}
#send-btn:hover {
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  box-shadow: 0 6px 24px rgba(99,82,246,0.4) !important;
  transform: translateY(-1px) !important;
}
#send-btn:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(67,56,202,0.3) !important;
}

/* ── Discovery Gallery (right panel) ─────────────────────────────────────── */
#html-panel {
  flex: 1;
  overflow-y: auto;
  padding: 12px 4px 20px;
  min-height: 0;

  /* Custom hidden scrollbar */
  scrollbar-width: thin;
  scrollbar-color: rgba(99,82,246,0.2) transparent;
}
#html-panel::-webkit-scrollbar       { width: 4px; }
#html-panel::-webkit-scrollbar-track { background: transparent; }
#html-panel::-webkit-scrollbar-thumb { background: rgba(99,82,246,0.2); border-radius: 99px; }

/* Gallery label */
#right-panel .label-wrap,
#html-panel > .label-wrap {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* ── Product Grid ─────────────────────────────────────────────────────────── */
.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 14px;
  padding: 4px 2px 8px;
}

/* ── Product Card (Glassmorphism) ─────────────────────────────────────────── */
.product-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 15px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: border-color var(--transition), transform var(--transition), box-shadow var(--transition);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  animation: cardIn 0.4s cubic-bezier(0.4,0,0.2,1) both;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}
.product-card:hover {
  border-color: rgba(139,92,246,0.35);
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(139,92,246,0.2), inset 0 1px 0 rgba(255,255,255,0.08);
}

@keyframes cardIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Card Image (white bg for product pop) ────────────────────────────────── */
.card-img-wrap  { position: relative; flex-shrink: 0; }
.card-img {
  width: 100%;
  height: 160px;
  object-fit: contain;
  display: block;
  background: #ffffff;
  padding: 8px;
}
.card-img-fallback {
  width: 100%;
  height: 160px;
  background: rgba(99,82,246,0.04);
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── Card Overlays ────────────────────────────────────────────────────────── */
.index-badge {
  position: absolute; top: 8px; left: 8px;
  background: rgba(5,5,5,0.75);
  color: var(--text-secondary);
  font-family: 'DM Sans', sans-serif;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--border-subtle);
  letter-spacing: 0.05em;
  backdrop-filter: blur(4px);
}
.card-badge {
  position: absolute; top: 8px; right: 8px;
  background: rgba(99,82,246,0.8);
  color: #e0e7ff;
  font-family: 'DM Sans', sans-serif;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  letter-spacing: 0.04em;
  backdrop-filter: blur(4px);
}

/* ── Card Body ────────────────────────────────────────────────────────────── */
.card-body {
  padding: 12px 14px 8px;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.card-name {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.35;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.card-rating  {
  font-size: 12px;
  color: var(--amber-warm);
  margin: 0;
  letter-spacing: 0.05em;
}
.rating-num   { color: var(--text-muted); font-size: 11px; margin-left: 2px; }
.card-reviews { font-size: 11px; color: var(--text-muted); margin: 0; }
.card-price {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--indigo-bright);
  margin: 4px 0 0;
  letter-spacing: -0.01em;
}

/* ── Card Footer ──────────────────────────────────────────────────────────── */
.card-footer { padding: 0 14px 14px; }
.card-link {
  display: block;
  text-align: center;
  background: rgba(67,56,202,0.12);
  border: 1px solid rgba(99,82,246,0.25);
  color: var(--indigo-bright);
  font-family: 'DM Sans', sans-serif;
  font-size: 12px;
  font-weight: 500;
  padding: 8px 0;
  border-radius: var(--radius-btn);
  text-decoration: none;
  transition: all var(--transition);
  letter-spacing: 0.02em;
}
.card-link:hover {
  background: rgba(99,82,246,0.2);
  border-color: rgba(139,92,246,0.45);
  color: #c4b5fd;
  box-shadow: 0 4px 12px rgba(99,82,246,0.15);
}
.card-link-disabled { color: var(--text-ghost); cursor: not-allowed; }

/* ── Results Header ───────────────────────────────────────────────────────── */
.results-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 4px 2px 14px;
}
.results-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}
.results-count {
  font-size: 11px;
  color: var(--text-muted);
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-subtle);
  padding: 2px 10px;
  border-radius: var(--radius-pill);
  font-family: 'DM Sans', sans-serif;
}
.fallback-notice {
  font-size: 11px;
  color: var(--amber-warm);
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.25);
  padding: 2px 10px;
  border-radius: var(--radius-pill);
  font-family: 'DM Sans', sans-serif;
}

/* ── Signal Tags ──────────────────────────────────────────────────────────── */
.signal-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 14px;
}
.signal-label {
  font-family: 'DM Sans', sans-serif;
  font-size: 10px;
  color: var(--text-ghost);
  margin-right: 2px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
}
.signal-tag {
  font-size: 11px;
  color: #67e8f9;
  background: rgba(103,232,249,0.07);
  border: 1px solid rgba(103,232,249,0.2);
  padding: 2px 10px;
  border-radius: var(--radius-pill);
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
}

/* ── Empty / Welcome / Error States ──────────────────────────────────────── */
.empty-state, .welcome-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  text-align: center;
}
.empty-icon, .welcome-icon {
  font-size: 32px;
  margin-bottom: 16px;
  color: var(--text-ghost);
  opacity: 0.6;
}
.empty-title, .welcome-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-secondary);
  margin: 0 0 8px;
}
.empty-sub, .welcome-sub {
  font-size: 13px;
  color: var(--text-muted);
  line-height: 1.65;
  margin: 0;
  font-weight: 400;
}
.welcome-sub em {
  color: var(--indigo-bright);
  font-style: normal;
  font-weight: 500;
}
.error-msg {
  font-size: 13px;
  color: #f87171;
  padding: 14px;
  font-family: 'DM Sans', sans-serif;
}

/* ── Input Row styling ────────────────────────────────────────────────────── */
.input-row {
  padding: 12px 12px 14px;
  border-top: 1px solid var(--border-subtle);
  background: rgba(255,255,255,0.02);
}
.followup-row {
  padding: 8px 12px 4px;
  border-top: 1px solid var(--border-subtle);
  gap: 6px !important;
}

/* ── Global scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,82,246,0.18); border-radius: 99px; }

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width: 900px) {
  #left-panel, #right-panel { height: auto; }
  .product-grid { grid-template-columns: 1fr 1fr; }
  .header-title { font-size: 1.6rem; }
}
@media (max-width: 540px) {
  .product-grid { grid-template-columns: 1fr; }
  .header-title { font-size: 1.3rem; }
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INITIAL VALUES
# ══════════════════════════════════════════════════════════════════════════════

_WELCOME_HTML = """
<div class="welcome-state">
  <div class="welcome-icon">◎</div>
  <p class="welcome-title">Discovery Gallery</p>
  <p class="welcome-sub">
    Describe what you're looking for in the chat.<br>
    Try <em>"best noise-cancelling headphones"</em> or <em>"Sony vs Bose"</em>.
  </p>
</div>
"""

# Single source of truth: make_state() from chatbot.py.
_INITIAL_STATE: dict = {**make_state(), "request_id": None}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GRADIO LAYOUT (Nisa AI Style)
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Sans"),
    ),
    css=CUSTOM_CSS,
    title="Product Recommender — Deterministic Recommendations",
) as demo:

    # ── State ─────────────────────────────────────────────────────────────────
    chat_state = gr.State(value=_INITIAL_STATE)

    # ── App Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
      <p class="header-eyebrow">AI-Powered · Deterministic</p>
      <h1 class="header-title">Product Recommender doesn't just search —<br><em>it resonates.</em></h1>
      <p class="header-sub">Hybrid recommendations that understand context, not just keywords.</p>
    </div>
    """)

    with gr.Row(equal_height=False, elem_classes=["main-row"]):

        # ── LEFT PANEL — Conversation ─────────────────────────────────────────
        with gr.Column(scale=4, elem_id="left-panel"):

            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chat-panel",
                height=390,
                type="tuples",
                show_copy_button=False,
                bubble_full_width=False,
            )

            # Follow-up pill buttons
            with gr.Row(elem_classes=["followup-row"]):
                btn1 = gr.Button("", visible=False, size="sm", elem_classes=["followup-btn"])
                btn2 = gr.Button("", visible=False, size="sm", elem_classes=["followup-btn"])
                btn3 = gr.Button("", visible=False, size="sm", elem_classes=["followup-btn"])

            # Input area
            with gr.Row(elem_classes=["input-row"]):
                text_input = gr.Textbox(
                    placeholder="e.g. 'best noise cancelling headphones under ₹30000'",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                    elem_id="query-input",
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button(
                    "Send ↗",
                    variant="primary",
                    scale=1,
                    elem_id="send-btn",
                )

        # ── RIGHT PANEL — Discovery Gallery ──────────────────────────────────
        with gr.Column(scale=6, elem_id="right-panel"):
            html_panel = gr.HTML(
                value=_WELCOME_HTML,
                elem_id="html-panel",
                label="Discovery Gallery",
                show_label=True,
            )

    # ── Output list — strict 7-output contract ────────────────────────────────
    outputs_list = [chatbot, html_panel, btn1, btn2, btn3, chat_state, text_input]

    # ── Event bindings ────────────────────────────────────────────────────────
    text_input.submit(
        fn=submit_query,
        inputs=[text_input, chat_state, chatbot],
        outputs=outputs_list,
        show_progress="minimal",
        concurrency_limit=1,
    )

    send_btn.click(
        fn=submit_query,
        inputs=[text_input, chat_state, chatbot],
        outputs=outputs_list,
        show_progress="minimal",
        concurrency_limit=1,
    )

    btn1.click(
        fn=submit_query,
        inputs=[btn1, chat_state, chatbot],
        outputs=outputs_list,
        show_progress="minimal",
        concurrency_limit=1,
    )
    btn2.click(
        fn=submit_query,
        inputs=[btn2, chat_state, chatbot],
        outputs=outputs_list,
        show_progress="minimal",
        concurrency_limit=1,
    )
    btn3.click(
        fn=submit_query,
        inputs=[btn3, chat_state, chatbot],
        outputs=outputs_list,
        show_progress="minimal",
        concurrency_limit=1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )