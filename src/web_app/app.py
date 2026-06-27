import sys
import os
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import uuid
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.core.config import CONFIG
from src.workflow.graph import run_query
from src.agents.portfolio_agent import PortfolioAnalysisAgent, _analyze_portfolio
from src.agents.goal_planning_agent import project_goal
from src.utils.market_data import get_market_overview, get_stock_info, get_price_history

cfg = CONFIG["streamlit"]
st.set_page_config(
    page_title=cfg["page_title"],
    page_icon=cfg["page_icon"],
    layout=cfg["layout"],
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "portfolio_holdings" not in st.session_state:
    st.session_state.portfolio_holdings = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💰 AI Finance Assistant")
    st.markdown("*Your intelligent financial education companion*")
    st.divider()
    st.markdown("**Agents available:**")
    st.markdown("""
- 🎓 Finance Q&A
- 📊 Portfolio Analysis
- 📈 Market Analysis
- 🎯 Goal Planning
- 📰 News Synthesizer
- 🧾 Tax Education
""")
    st.divider()
    st.caption("⚠️ For educational purposes only. Not financial advice.")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_portfolio, tab_market, tab_goals = st.tabs([
    "💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals"
])


# ─────────────────────────── CHAT TAB ────────────────────────────────────────

with tab_chat:
    st.header("Chat with your Financial Assistant")

    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant" and msg.get("agent"):
                st.caption(f"Agent: {msg['agent'].replace('_', ' ').title()}")

    if prompt := st.chat_input("Ask anything about investing, stocks, taxes, or financial planning..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = run_query(
                    user_query=prompt,
                    session_id=st.session_state.session_id,
                    portfolio_data={
                        "holdings": st.session_state.portfolio_holdings
                    } if st.session_state.portfolio_holdings else None,
                )
            st.markdown(result["response"])
            st.caption(f"Agent: {result['agent_used'].replace('_', ' ').title()}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "agent": result["agent_used"],
        })


# ──────────────────────── PORTFOLIO TAB ──────────────────────────────────────

with tab_portfolio:
    st.header("Portfolio Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Add Holdings")
        with st.form("add_holding"):
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper()
            shares = st.number_input("Shares", min_value=0.001, value=1.0, step=0.1)
            price = st.number_input("Current Price ($)", min_value=0.01, value=100.0, step=0.01)
            asset_type = st.selectbox(
                "Asset Type",
                ["US Stock", "International Stock", "Bond/Fixed Income", "ETF", "REIT", "Crypto", "Cash"]
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
            metrics = _analyze_portfolio(holdings)

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

            if st.button("Analyze Portfolio with AI", type="primary"):
                with st.spinner("Analyzing..."):
                    result = run_query(
                        user_query="Please analyze my portfolio in detail.",
                        session_id=st.session_state.session_id,
                        portfolio_data={"holdings": holdings},
                    )
                st.markdown(result["response"])
        else:
            st.info("Add holdings on the left or load the sample portfolio to get started.")


# ──────────────────────── MARKET TAB ─────────────────────────────────────────

with tab_market:
    st.header("Market Overview")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()

    with st.spinner("Loading market data..."):
        overview = get_market_overview()

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
        lookup_ticker = st.text_input("Enter ticker symbol", placeholder="AAPL, TSLA, MSFT...", label_visibility="collapsed")
    with btn_col:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and lookup_ticker:
        with st.spinner(f"Fetching {lookup_ticker.upper()}..."):
            info = get_stock_info(lookup_ticker.upper())
        if info:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${info['current_price']:,.2f}")
            c2.metric("1-Month Change", f"{info['monthly_change_pct']:+.2f}%")
            if info.get("yearly_change_pct") is not None:
                c3.metric("1-Year Change", f"{info['yearly_change_pct']:+.2f}%")
            if info.get("pe_ratio"):
                c4.metric("P/E Ratio", f"{info['pe_ratio']:.1f}")

            hist = get_price_history(lookup_ticker.upper(), period="6mo")
            if hist is not None:
                fig = px.line(
                    hist,
                    y="Close",
                    title=f"{info['name']} — 6-Month Price",
                    labels={"Close": "Price ($)", "Date": ""},
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

            if st.button(f"Ask AI about {lookup_ticker.upper()}", type="primary"):
                with st.spinner("Analyzing..."):
                    result = run_query(
                        user_query=f"Tell me about {lookup_ticker.upper()} stock and what beginners should know about it.",
                        session_id=st.session_state.session_id,
                    )
                st.markdown(result["response"])
        else:
            st.error(f"Could not find data for '{lookup_ticker}'. Check the ticker symbol.")


# ──────────────────────── GOALS TAB ──────────────────────────────────────────

with tab_goals:
    st.header("Financial Goal Planner")

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
            projection = project_goal(
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

            # Growth chart
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

            if st.button("Get AI Advice on This Goal", type="primary"):
                with st.spinner("Getting AI advice..."):
                    result = run_query(
                        user_query=f"Help me plan for my '{goal_name}' goal of ${goal_amount:,.0f} in {years} years.",
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
