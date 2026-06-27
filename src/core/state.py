from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    agent_type: str
    rag_context: str
    portfolio_data: Optional[dict]
    market_data: Optional[dict]
    response: str
    session_id: str
    error: Optional[str]
