"""LangGraph wiring for the debate protocol (see docs/agents.md).

Financial/Sentiment/Macro always run in round 1 (static fan-out from START).
If the Critic disagrees and the round limit isn't reached, PM decides which
of the three to re-ask (`pm_select_next_agents_node`), and only those are
dispatched via `Send` — a static edge to the same subset wouldn't work since
the subset is only known at runtime.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.nodes import (
    critic_node,
    finalize_node,
    financial_agent_node,
    macro_agent_node,
    pm_select_next_agents_node,
    pm_synthesize_node,
    sentiment_agent_node,
)
from src.agents.state import DebateState

MAX_ROUNDS = 3


def route_after_critic(state: DebateState) -> str:
    feedback = state["critic_feedback"]
    if feedback.agree or state["round_number"] >= MAX_ROUNDS:
        return "finalize"
    return "pm_select_next_agents"


def route_to_agents(state: DebateState) -> list[Send]:
    return [Send(agent, state) for agent in state["active_agents"]]


def build_graph() -> StateGraph:
    graph = StateGraph(DebateState)

    graph.add_node("financial", financial_agent_node)
    graph.add_node("sentiment", sentiment_agent_node)
    graph.add_node("macro", macro_agent_node)
    graph.add_node("pm_synthesize", pm_synthesize_node)
    graph.add_node("critic", critic_node)
    graph.add_node("pm_select_next_agents", pm_select_next_agents_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "financial")
    graph.add_edge(START, "sentiment")
    graph.add_edge(START, "macro")

    graph.add_edge("financial", "pm_synthesize")
    graph.add_edge("sentiment", "pm_synthesize")
    graph.add_edge("macro", "pm_synthesize")

    graph.add_edge("pm_synthesize", "critic")
    graph.add_conditional_edges("critic", route_after_critic)
    graph.add_conditional_edges("pm_select_next_agents", route_to_agents)

    graph.add_edge("finalize", END)

    return graph


compiled_graph = build_graph().compile()
