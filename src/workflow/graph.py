import logging
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.core.state import AgentState
from src.workflow.classifier import classify_query
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.tax_agent import TaxEducationAgent

logger = logging.getLogger(__name__)

_agents = {
    "finance_qa": FinanceQAAgent(),
    "portfolio_analysis": PortfolioAnalysisAgent(),
    "market_analysis": MarketAnalysisAgent(),
    "goal_planning": GoalPlanningAgent(),
    "news_synthesizer": NewsSynthesizerAgent(),
    "tax_education": TaxEducationAgent(),
}


def _classify_node(state: AgentState) -> AgentState:
    agent_type = classify_query(state["user_query"])
    logger.info(f"Routing to agent: {agent_type}")
    return {**state, "agent_type": agent_type}


def _route(state: AgentState) -> str:
    return state.get("agent_type", "finance_qa")


def _make_agent_node(name: str):
    agent = _agents[name]

    def node(state: AgentState) -> AgentState:
        return agent.run(state)

    node.__name__ = name
    return node


def _append_message(state: AgentState) -> AgentState:
    response = state.get("response", "")
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=state["user_query"]))
    messages.append(AIMessage(content=response))
    return {**state, "messages": messages}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classifier", _classify_node)
    for name in _agents:
        graph.add_node(name, _make_agent_node(name))
    graph.add_node("append_message", _append_message)

    graph.set_entry_point("classifier")
    graph.add_conditional_edges("classifier", _route, {name: name for name in _agents})
    for name in _agents:
        graph.add_edge(name, "append_message")
    graph.add_edge("append_message", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(
    user_query: str,
    session_id: str | None = None,
    conversation_history: list | None = None,
    portfolio_data: dict | None = None,
    market_data: dict | None = None,
) -> dict:
    graph = get_graph()
    state: AgentState = {
        "messages": conversation_history or [],
        "user_query": user_query,
        "agent_type": "",
        "rag_context": "",
        "portfolio_data": portfolio_data,
        "market_data": market_data,
        "response": "",
        "session_id": session_id or str(uuid.uuid4()),
        "error": None,
    }
    result = graph.invoke(state)
    return {
        "response": result.get("response", ""),
        "agent_used": result.get("agent_type", "unknown"),
        "messages": result.get("messages", []),
        "error": result.get("error"),
    }
