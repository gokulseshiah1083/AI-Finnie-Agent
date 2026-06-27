from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.rag.knowledge_base import get_knowledge_base


class TaxEducationAgent(BaseAgent):
    name = "tax_education"
    description = "Explains tax concepts, account types, and tax-efficient investing strategies"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a tax education specialist helping beginners understand investment-related taxes. "
            "Explain tax concepts clearly: capital gains, dividends, tax-advantaged accounts (401k, IRA, Roth), "
            "tax-loss harvesting, and tax-efficient investing strategies. "
            "Always include the disclaimer that this is educational content, not personalized tax advice, "
            "and users should consult a CPA or tax professional for their specific situation. "
            "Use 2024 tax figures when discussing contribution limits and tax brackets. "
            "Focus on strategies that help investors minimize taxes legally (tax efficiency)."
        )

    def _process(self, state: AgentState) -> str:
        kb = get_knowledge_base()
        context = kb.retrieve(state["user_query"], category="tax")
        updated_state = {**state, "rag_context": context}
        return super()._process(updated_state)
