"""Unit tests for src/agents/graph.py.

The compiled graph is invoked with `llm_client.generate_structured` scripted
(`fake_llm`) — no real DB, no real LLM. The three runs between them take every
branch of `route_after_critic` (agreement, disagreement, round limit) and pin
down the execution behavior: fan-out/fan-in, the round limit, and the
Send-based re-ask of only the agents the PM selects.
"""

import pytest

from src.agents.graph import compiled_graph, route_to_agents

pytestmark = pytest.mark.usefixtures("no_real_db")


def test_route_to_agents_creates_one_send_per_active_agent(debate_state):
    state = debate_state(active_agents=["financial", "macro"])
    sends = route_to_agents(state)
    assert [s.node for s in sends] == ["financial", "macro"]
    assert all(s.arg is state for s in sends)


def test_graph_terminates_after_round_1_when_critic_agrees(debate_state, fake_llm):
    llm = fake_llm(critic_agrees_after=0)

    result = compiled_graph.invoke(debate_state())

    assert result["round_number"] == 1
    assert result["critic_feedback"].agree is True
    assert llm.counts.get("NextAgentsSelection", 0) == 0
    assert llm.counts["AgentReport"] == 3  # financial + sentiment + macro, once

    final = result["final_result"]
    assert final.rounds_used == 1
    assert final.critic_agreed is True


def test_graph_reasks_only_selected_agent_then_agrees(debate_state, fake_llm):
    llm = fake_llm(critic_agrees_after=1, selected_agents=["macro"])

    result = compiled_graph.invoke(debate_state())

    assert result["round_number"] == 2
    assert result["active_agents"] == ["macro"]
    # 3 in round 1 + only 1 re-ask (macro) in round 2 — NOT 6.
    assert llm.counts["AgentReport"] == 4
    assert llm.counts["CriticFeedback"] == 2
    assert llm.counts["NextAgentsSelection"] == 1

    final = result["final_result"]
    assert final.rounds_used == 2
    assert final.critic_agreed is True


def test_graph_stops_at_round_limit_even_if_critic_keeps_disagreeing(
    debate_state, fake_llm
):
    llm = fake_llm(critic_agrees_after=99, selected_agents=["macro"])

    result = compiled_graph.invoke(debate_state())

    assert result["round_number"] == 3
    assert llm.counts["CriticFeedback"] == 3
    assert llm.counts["AgentReport"] == 5  # 3 + 1 (round 2) + 1 (round 3)

    final = result["final_result"]
    assert final.rounds_used == 3
    assert final.critic_agreed is False
