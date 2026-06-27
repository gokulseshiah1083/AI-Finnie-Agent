from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.rag.knowledge_base import get_knowledge_base


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    description = "Answers general financial education questions using the knowledge base"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a knowledgeable financial educator helping beginners understand investing and personal finance. "
            "Use the provided context from the knowledge base to give accurate, clear, and jargon-free explanations. "
            "Break down complex concepts into simple terms. Use examples and analogies. "
            "Always cite your sources when using context information. "
            "Be encouraging and empathetic — many users are new to investing and may feel intimidated."
        )

    def _process(self, state: AgentState) -> str:
        kb = get_knowledge_base()
        context = kb.retrieve(state["user_query"])
        updated_state = {**state, "rag_context": context}
        return super()._process(updated_state)
