"""Unit tests for src/agents/graph.py.

`route_after_critic` / `route_to_agents` are pure and tested directly. The
full-graph tests invoke the compiled graph with `llm_client.generate_structured`
monkeypatched — no real DB, no real LLM — to lock in the actual execution
behavior (fan-out/fan-in, the round-limit, and the Send-based re-ask of only
the agents the PM selects), matching what was verified manually before.
"""

from datetime import date

import pytest

from src.agents import llm_client, nodes
from src.agents.graph import compiled_graph, route_after_critic, route_to_agents
from src.agents.state import (
    AgentReport,
    CriticFeedback,
    DebateState,
    NextAgentsSelection,
    PMOpinion,
)


class _FakeSessionCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """None of these tests should ever touch Postgres — CI has no DB service."""
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_fundamentals", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_prices", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_macro_series", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_recent_articles", lambda *a, **k: [])


def _initial_state(**overrides) -> DebateState:
    state: DebateState = {
        "ticker": "AAPL",
        "horizon": "1W",
        "as_of_date": date(2025, 12, 30),
        "round_number": 1,
        "active_agents": ["financial", "sentiment", "macro"],
        "financial_report": None,
        "sentiment_report": None,
        "macro_report": None,
        "pm_opinion": None,
        "critic_feedback": None,
        "final_result": None,
    }
    state.update(overrides)
    return state


def _make_fake_llm(
    *, critic_agrees_after: int, selected_agents: list[str] | None = None
):
    """critic_agrees_after=N: the critic disagrees on its first N calls, then
    agrees from call N+1 onward (0 means it agrees immediately)."""
    counts: dict[str, int] = {}

    def fake(prompt, output_model):
        name = output_model.__name__
        counts[name] = counts.get(name, 0) + 1
        if output_model is AgentReport:
            return AgentReport(opinion="op", reasoning="rsn")
        if output_model is PMOpinion:
            return PMOpinion(
                direction="BUY", confidence=0.6, summary="s", arguments=["a"]
            )
        if output_model is CriticFeedback:
            agree = counts["CriticFeedback"] > critic_agrees_after
            return CriticFeedback(agree=agree, feedback="fb")
        if output_model is NextAgentsSelection:
            return NextAgentsSelection(agents=selected_agents or ["macro"])
        raise AssertionError(f"unexpected model {output_model}")

    fake.counts = counts
    return fake


# --------------------------------------------------------------------------- #
# route_after_critic / route_to_agents — pure, no graph execution needed
# --------------------------------------------------------------------------- #
def test_route_after_critic_finalizes_on_agreement():
    state = _initial_state(
        round_number=1, critic_feedback=CriticFeedback(agree=True, feedback="ok")
    )
    assert route_after_critic(state) == "finalize"


def test_route_after_critic_continues_on_disagreement_within_round_limit():
    state = _initial_state(
        round_number=1, critic_feedback=CriticFeedback(agree=False, feedback="no")
    )
    assert route_after_critic(state) == "pm_select_next_agents"


def test_route_after_critic_finalizes_at_round_limit_regardless_of_agreement():
    state = _initial_state(
        round_number=3, critic_feedback=CriticFeedback(agree=False, feedback="no")
    )
    assert route_after_critic(state) == "finalize"


def test_route_to_agents_creates_one_send_per_active_agent():
    state = _initial_state(active_agents=["financial", "macro"])
    sends = route_to_agents(state)
    assert [s.node for s in sends] == ["financial", "macro"]
    assert all(s.arg is state for s in sends)


# --------------------------------------------------------------------------- #
# Full graph — mocked LLM, no DB, no real API calls
# --------------------------------------------------------------------------- #
def test_graph_terminates_after_round_1_when_critic_agrees(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=0)
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    result = compiled_graph.invoke(_initial_state())

    assert result["round_number"] == 1
    assert result["critic_feedback"].agree is True
    assert fake_llm.counts.get("NextAgentsSelection", 0) == 0
    assert fake_llm.counts["AgentReport"] == 3  # financial + sentiment + macro, once

    final = result["final_result"]
    assert final.rounds_used == 1
    assert final.critic_agreed is True


def test_graph_reasks_only_selected_agent_then_agrees(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=1, selected_agents=["macro"])
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    result = compiled_graph.invoke(_initial_state())

    assert result["round_number"] == 2
    assert result["active_agents"] == ["macro"]
    # 3 in round 1 + only 1 re-ask (macro) in round 2 — NOT 6.
    assert fake_llm.counts["AgentReport"] == 4
    assert fake_llm.counts["CriticFeedback"] == 2
    assert fake_llm.counts["NextAgentsSelection"] == 1

    final = result["final_result"]
    assert final.rounds_used == 2
    assert final.critic_agreed is True


def test_graph_stops_at_round_limit_even_if_critic_keeps_disagreeing(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=99, selected_agents=["macro"])
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    result = compiled_graph.invoke(_initial_state())

    assert result["round_number"] == 3
    assert fake_llm.counts["CriticFeedback"] == 3
    assert fake_llm.counts["AgentReport"] == 5  # 3 + 1 (round 2) + 1 (round 3)

    final = result["final_result"]
    assert final.rounds_used == 3
    assert final.critic_agreed is False
