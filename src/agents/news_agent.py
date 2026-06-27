import logging
import yfinance as yf
from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.core.llm_client import call_llm

logger = logging.getLogger(__name__)


def _fetch_news(tickers: list[str] | None = None) -> list[dict]:
    symbols = tickers or ["SPY", "QQQ", "^GSPC"]
    all_news = []
    seen_titles = set()

    for sym in symbols[:3]:
        try:
            t = yf.Ticker(sym)
            news = t.news or []
            for item in news[:5]:
                title = item.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_news.append({
                        "title": title,
                        "publisher": item.get("publisher", ""),
                        "summary": item.get("summary", ""),
                        "link": item.get("link", ""),
                    })
        except Exception as e:
            logger.warning(f"Could not fetch news for {sym}: {e}")

    return all_news[:10]


class NewsSynthesizerAgent(BaseAgent):
    name = "news_synthesizer"
    description = "Summarizes and contextualizes financial news for beginner investors"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial news analyst who explains market news in plain English for beginner investors. "
            "For each news item: summarize what happened, explain why it matters, and what it might mean for investors. "
            "Avoid jargon. Contextualize numbers and events relative to historical norms. "
            "Help users understand the difference between short-term noise and long-term trends. "
            "Remind users that news-driven trading is often counterproductive for long-term investors."
        )

    def _process(self, state: AgentState) -> str:
        query = state["user_query"]
        history = self._extract_history(state)

        news_items = _fetch_news()
        if news_items:
            news_text = "\n\n".join(
                f"**{n['title']}** ({n['publisher']})\n{n.get('summary', 'No summary available')}"
                for n in news_items
            )
            user_msg = f"Recent Financial News:\n{news_text}\n\nUser Question: {query}"
        else:
            user_msg = (
                "Live news is temporarily unavailable. Answer the user's news-related question "
                f"using your general knowledge: {query}"
            )

        return call_llm(self.system_prompt, user_msg, history)
