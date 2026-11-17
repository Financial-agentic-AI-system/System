"""Fixtures shared by the unit tests: no real database, Redis or LLM."""

import sys
from datetime import date
from pathlib import Path

import fakeredis
import pytest

from src.agents import llm_client, nodes
from src.agents.state import (
    AgentReport,
    CriticFeedback,
    DebateState,
    NextAgentsSelection,
    PMOpinion,
)
from src.cache import debate_cache

# The frontend is a separate app that does not ship `src/`; its modules are
# imported by bare name (`import api_client`), as they are inside its container.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))


class FakeSessionCM:
    """Stands in for `get_session()`: a context manager yielding no session."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def fake_redis(monkeypatch):
    """debate_cache imports `redis_client` by name, so patch it there."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(debate_cache, "redis_client", fake)
    return fake


@pytest.fixture
def no_real_db(monkeypatch):
    """Every query the nodes make returns no rows — CI has no DB service.

    The Sentiment agent may ask the model for a search: `generate_tool_calls`
    answers "no search needed", so neither the endpoint nor the retriever runs.
    """
    monkeypatch.setattr(nodes, "get_session", lambda: FakeSessionCM())
    for fetch in (
        "fetch_fundamentals",
        "fetch_prices",
        "fetch_macro_series",
        "fetch_recent_articles",
        "fetch_articles_by_ids",
    ):
        monkeypatch.setattr(nodes, fetch, lambda *a, **k: [])
    monkeypatch.setattr(
        llm_client, "generate_tool_calls", lambda prompt, tools: [], raising=False
    )


@pytest.fixture
def debate_state():
    """Factory for a round-1 `DebateState`; keyword arguments override keys."""

    def make(**overrides) -> DebateState:
        state: DebateState = {
            "ticker": "AAPL",
            "horizon": "1W",
            "as_of_date": date(2025, 12, 30),
            "round_number": 1,
            "active_agents": ["financial", "sentiment", "macro"],
            "financial_report": None,
            "sentiment_report": None,
            "macro_report": None,
            "sentiment_queries": [],
            "sentiment_sources": [],
            "pm_opinion": None,
            "critic_feedback": None,
            "final_result": None,
        }
        state.update(overrides)
        return state

    return make


@pytest.fixture
def fake_llm(monkeypatch):
    """Factory that installs a scripted `llm_client.generate_structured`.

    critic_agrees_after=N: the critic disagrees on its first N calls, then
    agrees (0 means it agrees immediately). `fail_on=(schema_name, call_number)`
    makes that one call raise. The returned fake counts calls per schema in
    `.counts`.
    """

    def install(
        *,
        critic_agrees_after: int = 0,
        selected_agents: list[str] | None = None,
        fail_on: tuple[str, int] | None = None,
    ):
        counts: dict[str, int] = {}

        def fake(prompt, output_model):
            name = output_model.__name__
            counts[name] = counts.get(name, 0) + 1
            if (name, counts[name]) == fail_on:
                raise RuntimeError("simulated permanent LLM failure")
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
        monkeypatch.setattr(llm_client, "generate_structured", fake, raising=False)
        return fake

    return install
