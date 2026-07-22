"""Streamlit UI for AI Finnie Agent.

Styled as a finance service desk, sharing the design system of the companion
AI-Billing-Assistance project: config-driven branding, gradient section
banners, a segmented-control navigation, a pinned newest-first chat, and a
"⚡ Load AI engines now" button at the top of the page.

IMPORTANT — why this file is structured the way it is:

The front page must load INSTANTLY. So the top of this module imports only
light libraries (streamlit / pandas / plotly). The heavy backend — LangGraph,
the six agents, sentence-transformers (which drags in torch), FAISS, and the
market-data/yfinance stack — is imported LAZILY, only when the user presses
the preload button (or triggers an AI action). That is what keeps the home
page from hanging or showing a blank screen on Streamlit Community Cloud.

Every boot / load milestone is logged to stdout, and every query turn writes a
span tree to logs/trace.jsonl (see src/core/trace.py) — visible in the
Developer traces expander on the Welcome Desk.
"""
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on the path (needed when run directly via run.py)
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.core.config import CONFIG
from src.core.trace import read_trace_entries, tracing_enabled
from src.utils.logging_config import get_logger, boot
from src.web_app.components import render_label_css, render_section_header

log = get_logger("web_app")

_BRANDING = CONFIG["streamlit"].get("branding", {})
_LAZY = CONFIG["streamlit"].get("lazy_load", {})


# ── set_page_config (guarded) ─────────────────────────────────────────────────
def _safe_set_page_config():
    """The outer entry point (app.py) usually calls set_page_config first.
    When this module is run directly (python run.py / streamlit run src/web_app/app.py)
    nobody has called it yet, so we do — and swallow the error if it was already set."""
    cfg = CONFIG["streamlit"]
    try:
        st.set_page_config(
            page_title=cfg["page_title"],
            page_icon=cfg["page_icon"],
            layout=cfg["layout"],
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # already configured by the outer entry point


# ══════════════════════════════════════════════════════════════════════════════
#  LAZY BACKEND LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_ai_backend() -> dict:
    """Import + warm the heavy backend exactly ONCE per server process.

    st.cache_resource guarantees this runs a single time and the result is
    shared across reruns/sessions. Each step is logged so a slow or failing
    stage is obvious in the hosted logs.
    """
    t0 = time.time()
    boot("load_ai_backend: START")

    boot("load_ai_backend: importing workflow graph (langgraph + agents)...")
    from src.workflow.graph import run_query, get_graph
    boot(f"load_ai_backend: graph module imported (+{time.time() - t0:.1f}s)")

    boot("load_ai_backend: compiling LangGraph state machine...")
    get_graph()
    boot(f"load_ai_backend: graph compiled (+{time.time() - t0:.1f}s)")

    boot("load_ai_backend: warming knowledge base "
         "(loads sentence-transformers/torch + builds/loads FAISS index)...")
    from src.rag.knowledge_base import get_knowledge_base
    get_knowledge_base()
    boot(f"load_ai_backend: knowledge base ready (+{time.time() - t0:.1f}s)")

    boot("load_ai_backend: importing market-data + calculators...")
    from src.utils.market_data import get_market_overview, get_stock_info, get_price_history
    from src.agents.portfolio_agent import _analyze_portfolio
    from src.agents.goal_planning_agent import project_goal

    boot(f"load_ai_backend: DONE in {time.time() - t0:.1f}s")
    return {
        "run_query": run_query,
        "get_market_overview": get_market_overview,
        "get_stock_info": get_stock_info,
        "get_price_history": get_price_history,
        "analyze_portfolio": _analyze_portfolio,
        "project_goal": project_goal,
    }


def ensure_backend() -> dict:
    """Return the loaded backend, loading it (with a visible spinner) if needed."""
    if not st.session_state.get("backend_ready"):
        log.info("Backend not ready — loading now (user action or button).")
        with st.spinner("🔧 Loading the AI backend… first run downloads the "
                        "embedding model, so this can take 30–90s. Watch the logs."):
            backend = _load_ai_backend()
        st.session_state.backend_ready = True
        log.info("Backend marked ready.")
        return backend
    return _load_ai_backend()


# ── Light-weight calculators (no torch, safe to import any time) ───────────────
def _analyze_portfolio_light(holdings: list[dict]) -> dict:
    from src.agents.portfolio_agent import _analyze_portfolio
    return _analyze_portfolio(holdings)


def _project_goal_light(**kwargs) -> dict:
    from src.agents.goal_planning_agent import project_goal
    return project_goal(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "portfolio_holdings" not in st.session_state:
        st.session_state.portfolio_holdings = []
    if "backend_ready" not in st.session_state:
        st.session_state.backend_ready = False


def _render_sidebar():
    with st.sidebar:
        # Identity block: the assistant presented as a named analyst.
        st.markdown(
            f"## {_BRANDING.get('sidebar_icon', '🕵️')} "
            f"{_BRANDING.get('sidebar_title', 'AI Finnie Agent')}"
        )
        st.caption(_BRANDING.get("sidebar_role", "Personal Finance Analyst"))

        if st.button("🔄 New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.subheader("Agents available")
        for agent in _BRANDING.get("sidebar_agents", []):
            st.markdown(f"{agent.get('icon', '•')} **{agent.get('name', '')}**")
            if agent.get("blurb"):
                st.caption(agent["blurb"])

        st.divider()
        st.caption("⚠️ For educational purposes only. Not financial advice.")


def _backend_gate(feature: str) -> bool:
    """Show a friendly prompt when the heavy backend isn't loaded yet.
    Returns True if the backend is ready to use."""
    if st.session_state.backend_ready:
        return True
    st.info(f"⚡ **{feature}** needs the AI engines. Press **⚡ Load AI engines "
            "now** at the top of the page (or the button below) to get started.")
    if st.button(f"⚡ Load AI engines to enable {feature}", key=f"load_{feature}"):
        ensure_backend()
        st.rerun()
    return False


def _render_home_tab():
    render_section_header(
        "Welcome to your finance desk",
        "Finnie is a team of six specialist agents for money questions, "
        "portfolios, markets, goals, news and taxes — in plain English.",
        "🏠",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Works instantly")
        st.markdown(
            "- 📊 **My Portfolio** metrics & charts (pure math)\n"
            "- 🎯 **Goal Planner** projections & charts (pure math)\n"
        )
    with c2:
        st.subheader("Needs the AI engines")
        st.markdown(
            "- 💬 **Ask Finnie** — chat with the agents\n"
            "- 📈 **Market Watch** — live market data\n"
            "- 🤖 Any *“Analyze / Ask AI”* button\n"
        )

    # Developer traces — the debugging window into every query turn.
    if tracing_enabled():
        with st.expander("🔬 Developer traces (span-level timing)", expanded=False):
            st.caption(
                "One row per completed span. Rows sharing a `trace_id` belong to "
                "the same query turn: turn → graph nodes → model calls. Full "
                "detail lives in `logs/trace.jsonl`."
            )
            traces = read_trace_entries(limit=200)
            if not traces:
                st.info("No traces yet — ask Finnie a question first.", icon="ℹ️")
            else:
                st.dataframe(
                    [
                        {
                            "ts": t.get("ts"),
                            "trace_id": t.get("trace_id"),
                            "span": t.get("name"),
                            "duration_ms": t.get("duration_ms"),
                            "status": t.get("status"),
                            "route": t.get("route") or t.get("agent_used"),
                            "model": t.get("model"),
                            "tokens_out": t.get("output_tokens"),
                            "error": t.get("error"),
                        }
                        for t in traces
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


def _render_chat_tab():
    render_section_header(
        "How can Finnie help you today?",
        "Ask about investing, your portfolio, markets, goals or taxes — in "
        "your own words. The right specialist agent picks it up automatically.",
        "💬",
    )

    # Popular requests: ALWAYS visible at the top, re-clickable after every
    # response — not just on an empty conversation.
    st.markdown("**Popular questions — tap one any time:**")
    examples = [
        "What's the difference between a Roth IRA and a traditional IRA?",
        "Analyze my portfolio and tell me what to improve",
        "How do capital gains taxes work when I sell a stock?",
        "Is now a reasonable time for a beginner to start investing?",
    ]
    columns = st.columns(2)
    for index, example in enumerate(examples):
        if columns[index % 2].button(example, key=f"ex_{index}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

    # The input box lives HERE, directly under the popular questions, and stays
    # here: history renders newest-first below it, so asking the next question
    # never requires scrolling.
    prompt = st.chat_input("Ask anything about investing, stocks, taxes, or financial planning…")
    if not prompt and "pending_query" in st.session_state:
        prompt = st.session_state.pop("pending_query")

    if prompt:
        log.info("Chat prompt received: %r", prompt[:120])
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            backend = ensure_backend()  # auto-loads on first use
            result = backend["run_query"](
                user_query=prompt,
                session_id=st.session_state.session_id,
                portfolio_data={"holdings": st.session_state.portfolio_holdings}
                if st.session_state.portfolio_holdings else None,
            )
        log.info("Chat answered by agent=%s", result.get("agent_used"))
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "agent": result["agent_used"],
        })
        # Fall through: the turn just appended renders first in the loop below.

    # History, newest turn first. Messages are grouped into turns (a user
    # message plus the assistant reply) so each Q&A stays in reading order
    # even though the turns themselves are reversed.
    turns: list[list[dict]] = []
    for msg in st.session_state.messages:
        if msg["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(msg)

    if turns:
        st.caption("Latest first — your newest answer is always right here at the top.")
    for turn in reversed(turns):
        for msg in turn:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("agent"):
                    st.caption(f"Agent: {msg['agent'].replace('_', ' ').title()}")
        st.divider()


def _render_portfolio_tab():
    render_section_header(
        "Your portfolio, at a glance",
        "Add holdings to see value, allocation and diversification instantly — "
        "pure math, no AI engines needed. AI review is one button away.",
        "📊",
    )
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Add Holdings")
        with st.form("add_holding"):
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper()
            shares = st.number_input("Shares", min_value=0.001, value=1.0, step=0.1)
            price = st.number_input("Current Price ($)", min_value=0.01, value=100.0, step=0.01)
            asset_type = st.selectbox(
                "Asset Type",
                ["US Stock", "International Stock", "Bond/Fixed Income", "ETF", "REIT", "Crypto", "Cash"],
            )
            submitted = st.form_submit_button("Add Holding", use_container_width=True)
            if submitted and ticker:
                st.session_state.portfolio_holdings.append({
                    "ticker": ticker,
                    "shares": shares,
                    "current_price": price,
                    "asset_type": asset_type,
                })
                st.success(f"Added {ticker}")
                st.rerun()

        if st.button("Load Sample Portfolio", use_container_width=True):
            st.session_state.portfolio_holdings = [
                {"ticker": "VTI", "shares": 50, "current_price": 240, "asset_type": "ETF"},
                {"ticker": "VXUS", "shares": 30, "current_price": 58, "asset_type": "ETF"},
                {"ticker": "BND", "shares": 20, "current_price": 73, "asset_type": "Bond/Fixed Income"},
                {"ticker": "AAPL", "shares": 10, "current_price": 195, "asset_type": "US Stock"},
                {"ticker": "MSFT", "shares": 5, "current_price": 415, "asset_type": "US Stock"},
            ]
            st.rerun()

        if st.session_state.portfolio_holdings and st.button("Clear Portfolio", use_container_width=True):
            st.session_state.portfolio_holdings = []
            st.rerun()

    with col2:
        holdings = st.session_state.portfolio_holdings
        if holdings:
            metrics = _analyze_portfolio_light(holdings)  # light math, no backend needed

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Value", f"${metrics['total_value']:,.2f}")
            m2.metric("Holdings", metrics["num_holdings"])
            m3.metric("Diversification Score", f"{metrics['diversification_score']}/100")

            st.subheader("Asset Allocation")
            alloc = metrics["asset_allocation"]
            fig_alloc = go.Figure(go.Pie(
                labels=list(alloc.keys()),
                values=list(alloc.values()),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_alloc.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=280)
            st.plotly_chart(fig_alloc, use_container_width=True)

            st.subheader("Holdings Breakdown")
            df = pd.DataFrame(metrics["holdings"])
            df["value"] = df["value"].apply(lambda x: f"${x:,.2f}")
            df["percentage"] = df["percentage"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df, use_container_width=True, hide_index=True)

            if st.button("🤖 Analyze Portfolio with AI", type="primary"):
                if _backend_gate("AI portfolio analysis"):
                    with st.spinner("Analyzing..."):
                        backend = ensure_backend()
                        result = backend["run_query"](
                            user_query="Please analyze my portfolio in detail.",
                            session_id=st.session_state.session_id,
                            portfolio_data={"holdings": holdings},
                        )
                    st.markdown(result["response"])
        else:
            st.info("Add holdings on the left or load the sample portfolio to get started.")


def _render_market_tab():
    render_section_header(
        "Markets, right now",
        "Live index levels, stock lookups and price history — with a "
        "plain-English AI explainer for any ticker.",
        "📈",
    )

    if not _backend_gate("Live market data"):
        return

    backend = ensure_backend()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            log.info("Market cache cleared by user.")

    with st.spinner("Loading market data..."):
        log.info("Fetching market overview...")
        overview = backend["get_market_overview"]()
        log.info("Market overview fetched: %d symbols", len(overview or {}))

    if overview:
        cols = st.columns(len(overview))
        for i, (sym, data) in enumerate(overview.items()):
            with cols[i]:
                change = data.get("monthly_change_pct", 0) or 0
                color = "normal" if change >= 0 else "inverse"
                st.metric(
                    label=sym,
                    value=f"${data['current_price']:,.2f}",
                    delta=f"{change:+.2f}% (1mo)",
                    delta_color=color,
                )

    st.divider()
    st.subheader("Stock Lookup")
    search_col, btn_col = st.columns([3, 1])
    with search_col:
        lookup_ticker = st.text_input(
            "Enter ticker symbol", placeholder="AAPL, TSLA, MSFT...", label_visibility="collapsed"
        )
    with btn_col:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and lookup_ticker:
        with st.spinner(f"Fetching {lookup_ticker.upper()}..."):
            log.info("Stock lookup: %s", lookup_ticker.upper())
            info = backend["get_stock_info"](lookup_ticker.upper())
        if info:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${info['current_price']:,.2f}")
            c2.metric("1-Month Change", f"{info['monthly_change_pct']:+.2f}%")
            if info.get("yearly_change_pct") is not None:
                c3.metric("1-Year Change", f"{info['yearly_change_pct']:+.2f}%")
            if info.get("pe_ratio"):
                c4.metric("P/E Ratio", f"{info['pe_ratio']:.1f}")

            hist = backend["get_price_history"](lookup_ticker.upper(), period="6mo")
            if hist is not None:
                fig = px.line(
                    hist, y="Close",
                    title=f"{info['name']} — 6-Month Price",
                    labels={"Close": "Price ($)", "Date": ""},
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

            if st.button(f"🤖 Ask AI about {lookup_ticker.upper()}", type="primary"):
                with st.spinner("Analyzing..."):
                    result = backend["run_query"](
                        user_query=f"Tell me about {lookup_ticker.upper()} stock and what "
                                   f"beginners should know about it.",
                        session_id=st.session_state.session_id,
                    )
                st.markdown(result["response"])
        else:
            st.error(f"Could not find data for '{lookup_ticker}'. Check the ticker symbol.")


def _render_goals_tab():
    render_section_header(
        "Plan the goal, see the path",
        "Set a target and see the projection instantly — how much, how long, "
        "and what it takes monthly. AI advice on top when you want it.",
        "🎯",
    )
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Set Your Goal")
        goal_name = st.text_input("Goal Name", placeholder="Retirement / Home Down Payment / Emergency Fund")
        goal_amount = st.number_input("Target Amount ($)", min_value=100.0, value=100000.0, step=1000.0)
        current_savings = st.number_input("Current Savings ($)", min_value=0.0, value=5000.0, step=100.0)
        monthly_contrib = st.number_input("Monthly Contribution ($)", min_value=0.0, value=500.0, step=50.0)
        annual_return = st.slider("Expected Annual Return (%)", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
        years = st.slider("Time Horizon (Years)", min_value=1, max_value=50, value=20)
        st.caption("Return estimates: Savings 4-5% | Bonds 5-6% | Balanced 7% | Stocks 9-10%")
        calc_btn = st.button("Calculate Projection", type="primary", use_container_width=True)

    with col2:
        if calc_btn:
            projection = _project_goal_light(  # light math, no backend needed
                goal_amount=goal_amount,
                current_savings=current_savings,
                monthly_contribution=monthly_contrib,
                annual_return_pct=annual_return,
                years=years,
            )

            status = "✅ On Track!" if projection["on_track"] else "⚠️ Adjustment Needed"
            st.subheader(f"{goal_name or 'Goal'} — {status}")

            m1, m2 = st.columns(2)
            m1.metric("Projected Total", f"${projection['projected_total']:,.0f}")
            m2.metric("Target", f"${goal_amount:,.0f}")

            if projection["shortfall"] > 0:
                st.warning(f"Shortfall: ${projection['shortfall']:,.0f}")
                st.info(f"Required monthly contribution: **${projection['required_monthly_contribution']:,.0f}**/month")
            else:
                surplus = projection['projected_total'] - goal_amount
                st.success(f"Surplus: ${surplus:,.0f} 🎉")

            if projection.get("years_to_goal_at_current_rate"):
                st.metric("Years to Goal at Current Rate", f"{projection['years_to_goal_at_current_rate']:.1f} years")

            months_range = range(0, years * 12 + 1, max(1, years // 20))
            r = annual_return / 100 / 12
            chart_data = []
            for m in months_range:
                if r > 0:
                    val = current_savings * (1 + r) ** m + monthly_contrib * (((1 + r) ** m - 1) / r)
                else:
                    val = current_savings + monthly_contrib * m
                chart_data.append({"Month": m, "Projected Value": val})

            df_chart = pd.DataFrame(chart_data)
            fig = px.area(
                df_chart, x="Month", y="Projected Value",
                title=f"Projected Growth Over {years} Years",
                labels={"Month": "Months", "Projected Value": "Value ($)"},
            )
            fig.add_hline(y=goal_amount, line_dash="dash", line_color="red",
                          annotation_text=f"Goal: ${goal_amount:,.0f}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🤖 Get AI Advice on This Goal", type="primary"):
                if _backend_gate("AI goal advice"):
                    with st.spinner("Getting AI advice..."):
                        backend = ensure_backend()
                        result = backend["run_query"](
                            user_query=f"Help me plan for my '{goal_name}' goal of "
                                       f"${goal_amount:,.0f} in {years} years.",
                            session_id=st.session_state.session_id,
                            market_data={"goal_params": {
                                "goal_amount": goal_amount,
                                "current_savings": current_savings,
                                "monthly_contribution": monthly_contrib,
                                "annual_return_pct": annual_return,
                                "years": years,
                            }},
                        )
                    st.markdown(result["response"])


def render():
    """Entry point used by the outer app.py. Renders the whole UI (light)."""
    boot("web_app.render(): building UI shell (no heavy imports here)")
    _init_session_state()
    _render_sidebar()
    render_label_css()

    st.title(
        f"{CONFIG['streamlit']['page_icon']} "
        f"{_BRANDING.get('app_name', CONFIG['streamlit']['page_title'])}"
    )
    st.caption(_BRANDING.get("tagline", "Your intelligent financial education companion."))

    # Engine status + on-demand loading, at the top of the page. The engines
    # load lazily on the first AI action; the button pulls that cost forward
    # at a moment the user chooses.
    if st.session_state.backend_ready:
        st.caption("✅ AI engines loaded — every view is fully enabled.")
    else:
        status_col, button_col = st.columns([3, 1])
        status_col.caption(
            "AI engines start on your first question — or load them now, "
            "at a moment you choose."
        )
        if _LAZY.get("enabled", True):
            if button_col.button(
                _LAZY.get("preload_button_label", "⚡ Load AI engines now"),
                use_container_width=True,
            ):
                log.info("User clicked the preload button.")
                ensure_backend()
                st.rerun()

    # Navigation is a segmented control (radio fallback) rather than st.tabs,
    # matching the companion billing project — and keeping the door open for
    # programmatic view switching via session state.
    labels = _BRANDING.get("tab_labels", {})
    views = {
        labels.get("home", "🏠 Welcome Desk"): _render_home_tab,
        labels.get("chat", "💬 Ask Finnie"): _render_chat_tab,
        labels.get("portfolio", "📊 My Portfolio"): _render_portfolio_tab,
        labels.get("market", "📈 Market Watch"): _render_market_tab,
        labels.get("goals", "🎯 Goal Planner"): _render_goals_tab,
    }
    if "pending_view" in st.session_state:
        st.session_state.nav_view = st.session_state.pop("pending_view")
    view_names = list(views)
    st.session_state.setdefault("nav_view", view_names[0])
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "View", view_names, key="nav_view", label_visibility="collapsed"
        )
    else:
        selected = st.radio(
            "View", view_names, key="nav_view", horizontal=True, label_visibility="collapsed"
        )
    views[selected or view_names[0]]()
    boot("web_app.render(): UI shell built")


# When run directly (python run.py -> streamlit run src/web_app/app.py)
if __name__ == "__main__":
    _safe_set_page_config()
    render()
