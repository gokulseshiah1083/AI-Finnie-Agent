import json
import math
import logging
from src.core.base_agent import BaseAgent
from src.core.state import AgentState
from src.core.llm_client import call_llm

logger = logging.getLogger(__name__)


def project_goal(
    goal_amount: float,
    current_savings: float,
    monthly_contribution: float,
    annual_return_pct: float,
    years: int,
) -> dict:
    r = annual_return_pct / 100 / 12
    months = years * 12

    if r > 0:
        future_value_savings = current_savings * (1 + r) ** months
        future_value_contributions = monthly_contribution * (((1 + r) ** months - 1) / r)
    else:
        future_value_savings = current_savings
        future_value_contributions = monthly_contribution * months

    projected_total = future_value_savings + future_value_contributions
    shortfall = max(0, goal_amount - projected_total)

    if shortfall > 0 and r > 0:
        required_monthly = (goal_amount - future_value_savings) * r / (((1 + r) ** months - 1))
    elif shortfall > 0:
        required_monthly = (goal_amount - current_savings) / months
    else:
        required_monthly = monthly_contribution

    years_to_goal_at_current = None
    if monthly_contribution > 0:
        try:
            if r > 0:
                n = math.log((goal_amount * r / monthly_contribution) + 1) / math.log(1 + r)
            else:
                n = (goal_amount - current_savings) / monthly_contribution
            years_to_goal_at_current = round(n / 12, 1)
        except Exception:
            pass

    return {
        "goal_amount": goal_amount,
        "projected_total": round(projected_total, 2),
        "shortfall": round(shortfall, 2),
        "on_track": shortfall == 0,
        "required_monthly_contribution": round(required_monthly, 2),
        "years_to_goal_at_current_rate": years_to_goal_at_current,
        "assumptions": {
            "annual_return_pct": annual_return_pct,
            "years": years,
            "current_savings": current_savings,
            "monthly_contribution": monthly_contribution,
        },
    }


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    description = "Helps users set and plan financial goals with projection calculations"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial goal planning coach helping users define and achieve their financial goals. "
            "When goal data is provided, interpret the projection results clearly: explain whether they're on track, "
            "how much they need to save monthly, and what adjustments could help. "
            "Ask clarifying questions if goal details are missing: What is the goal? How much? By when? "
            "What is their current savings and monthly contribution? What return rate is expected? "
            "Always explain assumptions and what changing them means. "
            "Common return assumptions: savings account 4-5%, conservative bonds 5%, balanced 7%, aggressive stocks 9-10%. "
            "Account for risk tolerance in return assumptions."
        )

    def _process(self, state: AgentState) -> str:
        query = state["user_query"]
        history = self._extract_history(state)

        goal_context = ""
        market_data = state.get("market_data")
        if not isinstance(market_data, dict):
            market_data = {}

        goal_data = market_data.get("goal_params")
        if isinstance(goal_data, dict):
            try:
                projection = project_goal(**goal_data)
                goal_context = f"Goal Projection Results:\n{json.dumps(projection, indent=2)}\n\n"
            except Exception as e:
                logger.error(f"Goal projection error: {e}")

        user_msg = f"{goal_context}User Question: {query}"
        return call_llm(self.system_prompt, user_msg, history)

    def calculate_projection(self, **kwargs) -> dict:
        return project_goal(**kwargs)
