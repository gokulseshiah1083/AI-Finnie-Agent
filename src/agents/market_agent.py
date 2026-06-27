import json
import logging
from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.core.llm_client import call_llm
from src.utils.market_data import get_stock_info, get_market_overview
import re

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')


def _extract_tickers(text: str) -> list[str]:
    common_words = {"I", "A", "THE", "AND", "OR", "FOR", "IN", "ON", "AT", "OF", "TO", "IS", "ARE",
                    "MY", "ME", "DO", "HOW", "CAN", "GET", "BUY", "SELL", "WHAT", "ETF", "US", "AI"}
    candidates = TICKER_PATTERN.findall(text.upper())
    return [c for c in candidates if c not in common_words]


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"
    description = "Provides real-time market data, stock analysis, and market trend insights"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a market analyst providing real-time market insights for beginner investors. "
            "Use the provided live market data to explain what's happening in the market. "
            "Explain financial metrics (P/E ratio, market cap, dividend yield, etc.) in simple terms. "
            "Discuss trends without making specific buy/sell recommendations. "
            "Help users understand what market data means for their investment education. "
            "Always contextualize numbers (e.g., 'a P/E of 25 is above the historical average of ~15-17')."
        )

    def _process(self, state: AgentState) -> str:
        query = state["user_query"]
        history = self._extract_history(state)

        tickers = _extract_tickers(query)
        market_data = {}

        if tickers:
            for ticker in tickers[:5]:
                info = get_stock_info(ticker)
                if info:
                    market_data[ticker] = info

        if not market_data:
            market_data = get_market_overview()

        data_str = json.dumps(market_data, indent=2, default=str)
        user_msg = f"Live Market Data:\n{data_str}\n\nUser Question: {query}"
        return call_llm(self.system_prompt, user_msg, history)
