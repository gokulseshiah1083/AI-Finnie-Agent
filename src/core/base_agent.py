import logging
from abc import ABC, abstractmethod
from src.core.state import AgentState
from src.core.llm_client import call_llm
from src.core.config import CONFIG

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "\n\n---\n*This information is for educational purposes only and does not constitute "
    "financial advice. Always consult a qualified financial advisor before making investment decisions.*"
)


class BaseAgent(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self):
        self.config = CONFIG

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def run(self, state: AgentState) -> AgentState:
        try:
            response = self._process(state)
            return {**state, "response": response + DISCLAIMER, "error": None}
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            fallback = self.config["agents"]["fallback_message"]
            return {**state, "response": fallback, "error": str(e)}

    def _process(self, state: AgentState) -> str:
        history = self._extract_history(state)
        user_msg = self._build_user_message(state)
        return call_llm(self.system_prompt, user_msg, history)

    def _extract_history(self, state: AgentState) -> list:
        history = []
        for msg in state.get("messages", []):
            role = getattr(msg, "type", None)
            content = getattr(msg, "content", "")
            if role == "human":
                history.append({"role": "user", "content": content})
            elif role == "ai":
                history.append({"role": "assistant", "content": content})
        # Keep last 10 turns
        return history[-20:]

    def _build_user_message(self, state: AgentState) -> str:
        query = state["user_query"]
        context = state.get("rag_context", "")
        if context:
            return f"Context from knowledge base:\n{context}\n\nUser question: {query}"
        return query
