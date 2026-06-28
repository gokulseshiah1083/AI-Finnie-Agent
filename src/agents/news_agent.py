import logging
from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.core.llm_client import call_llm
from src.utils.market_data import get_financial_news

logger = logging.getLogger(__name__)


class NewsSynthesizerAgent(BaseAgent):
    name = "news_synthesizer"
    description = "Summarizes and contextualizes financial news for beginner investors"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial news analyst who explains market news in plain English for beginner investors. "
            "For each news item: summarize what happened, explain why it matters, and what it might mean for investors. "
            "When sentiment data is available (Bullish/Bearish/Neutral), explain what it means. "
            "Avoid jargon. Contextualize numbers and events relative to historical norms. "
            "Help users understand the difference between short-term noise and long-term trends. "
            "Remind users that news-driven trading is often counterproductive for long-term investors."
        )

    def _process(self, state: AgentState) -> str:
        query = state["user_query"]
        history = self._extract_history(state)

        news_items = get_financial_news()
        if news_items:
            news_text = "\n\n".join(
                self._format_news_item(n) for n in news_items
            )
            user_msg = f"Recent Financial News (via Alpha Vantage):\n{news_text}\n\nUser Question: {query}"
        else:
            user_msg = (
                "Live news is temporarily unavailable. Answer the user's news-related question "
                f"using your general knowledge: {query}"
            )

        return call_llm(self.system_prompt, user_msg, history)

    def _format_news_item(self, n: dict) -> str:
        sentiment = f" | Sentiment: {n['sentiment']}" if n.get("sentiment") else ""
        return (
            f"**{n['title']}** ({n['publisher']}){sentiment}\n"
            f"{n.get('summary', 'No summary available')}"
        )
