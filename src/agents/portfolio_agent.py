import json
import logging
from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.core.llm_client import call_llm

logger = logging.getLogger(__name__)


def _analyze_portfolio(holdings: list[dict]) -> dict:
    """Compute portfolio metrics from a list of {ticker, shares, current_price, asset_type}."""
    if not holdings:
        return {}

    total_value = sum(h.get("shares", 0) * h.get("current_price", 0) for h in holdings)
    if total_value == 0:
        return {}

    allocations = []
    asset_type_totals: dict[str, float] = {}
    for h in holdings:
        value = h.get("shares", 0) * h.get("current_price", 0)
        pct = (value / total_value) * 100
        allocations.append({
            "ticker": h.get("ticker"),
            "value": round(value, 2),
            "percentage": round(pct, 2),
            "asset_type": h.get("asset_type", "unknown"),
        })
        asset_type = h.get("asset_type", "other")
        asset_type_totals[asset_type] = asset_type_totals.get(asset_type, 0) + value

    # Diversification score (0-100): penalize concentration
    max_pct = max(a["percentage"] for a in allocations) if allocations else 100
    n_holdings = len(allocations)
    concentration_penalty = max(0, max_pct - 30) * 0.5
    size_bonus = min(n_holdings * 5, 30)
    diversification_score = max(0, min(100, 70 - concentration_penalty + size_bonus))

    asset_allocation = {k: round((v / total_value) * 100, 2) for k, v in asset_type_totals.items()}

    return {
        "total_value": round(total_value, 2),
        "holdings": allocations,
        "asset_allocation": asset_allocation,
        "diversification_score": round(diversification_score, 1),
        "num_holdings": n_holdings,
    }


class PortfolioAnalysisAgent(BaseAgent):
    name = "portfolio_analysis"
    description = "Analyzes user portfolio for diversification, allocation, and risk metrics"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a portfolio analyst helping beginner investors understand their investment holdings. "
            "Analyze the provided portfolio metrics and give clear, actionable insights about: "
            "1) Overall portfolio composition and diversification, "
            "2) Risk level based on asset allocation, "
            "3) Concentration risks (any single holding > 20% of portfolio), "
            "4) Gaps in diversification, "
            "5) Specific, beginner-friendly recommendations. "
            "Use simple language. Explain what each metric means before interpreting it."
        )

    def _process(self, state: AgentState) -> str:
        portfolio_data = state.get("portfolio_data") or {}
        holdings = portfolio_data.get("holdings", [])

        metrics = _analyze_portfolio(holdings)
        query = state["user_query"]
        history = self._extract_history(state)

        if metrics:
            portfolio_summary = json.dumps(metrics, indent=2)
            user_msg = (
                f"Portfolio Analysis Data:\n{portfolio_summary}\n\n"
                f"User Question: {query}"
            )
        else:
            user_msg = (
                f"The user has not provided portfolio data yet. "
                f"Ask them to enter their holdings (ticker symbol, number of shares, current price, asset type). "
                f"User Question: {query}"
            )

        return call_llm(self.system_prompt, user_msg, history)

    def get_metrics(self, holdings: list[dict]) -> dict:
        return _analyze_portfolio(holdings)
