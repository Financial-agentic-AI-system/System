"""Unit tests for src/worker/tasks.py — the Celery task that runs the debate
graph and mirrors its progress into src/cache/debate_cache.py.

Combines fixtures already established elsewhere: `_no_real_db` and
`_make_fake_llm` (tests/unit/test_agents_graph.py) and `_fake_redis`
(tests/unit/test_debate_cache.py). Tests call `_run_debate` directly — no
Celery machinery involved — except one test that goes through the
`@app.task` wrapper via `.apply()`, to prove the wrapping itself works.
"""

from datetime import date

import fakeredis
import pytest

from src.agents import llm_client, nodes
from src.agents.state import (
    AgentReport,
    CriticFeedback,
    NextAgentsSelection,
    PMOpinion,
)
from src.cache import debate_cache
from src.worker import tasks


class _FakeSessionCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_fundamentals", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_prices", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_macro_series", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_recent_articles", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(debate_cache, "redis_client", fake)
    return fake


def _make_fake_llm(
    *,
    critic_agrees_after: int,
    selected_agents: list[str] | None = None,
    fail_on: tuple[str, int] | None = None,
):
    """Same shape as test_agents_graph.py's helper, plus an optional
    (schema_name, call_number) pair that makes that one call raise instead
    of returning — used to exercise the FAILED status path.
    """
    counts: dict[str, int] = {}

    def fake(prompt, output_model):
        name = output_model.__name__
        counts[name] = counts.get(name, 0) + 1
        if fail_on is not None and (name, counts[name]) == fail_on:
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
    return fake


def test_run_debate_happy_path_writes_done_status_result_and_history(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=0)
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    result = tasks._run_debate("AAPL", "1W", date(2025, 12, 30), task_id="t1")

    assert result.direction == "BUY"
    assert result.rounds_used == 1

    status = debate_cache.get_status("t1")
    assert status.status == "DONE"
    assert status.round_number == 1
    assert status.current_node == "finalize"

    fetched_result = debate_cache.get_result("t1")
    assert fetched_result.direction == "BUY"

    history = debate_cache.get_history("t1")
    # financial/sentiment/macro run in parallel — order among the three
    # isn't guaranteed, only that all three land before the sequential tail.
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert [h.node for h in history[3:]] == ["pm_synthesize", "critic", "finalize"]
    assert all(h.round_number == 1 for h in history)


def test_run_debate_partial_reask_tags_round_2_entries(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=1, selected_agents=["macro"])
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    result = tasks._run_debate("AAPL", "1W", date(2025, 12, 30), task_id="t1")

    assert result.rounds_used == 2

    history = debate_cache.get_history("t1")
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert [h.node for h in history[3:]] == [
        "pm_synthesize",
        "critic",
        "pm_select_next_agents",
        "macro",
        "pm_synthesize",
        "critic",
        "finalize",
    ]
    assert [h.round_number for h in history] == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    status = debate_cache.get_status("t1")
    assert status.round_number == 2


def test_run_debate_marks_failed_on_exception_and_preserves_partial_history(
    monkeypatch,
):
    fake_llm = _make_fake_llm(critic_agrees_after=0, fail_on=("CriticFeedback", 1))
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    with pytest.raises(RuntimeError):
        tasks._run_debate("AAPL", "1W", date(2025, 12, 30), task_id="t1")

    status = debate_cache.get_status("t1")
    assert status.status == "FAILED"
    assert status.error is not None

    # financial/sentiment/macro/pm_synthesize made it in before critic raised
    history = debate_cache.get_history("t1")
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert [h.node for h in history[3:]] == ["pm_synthesize"]

    assert debate_cache.get_result("t1") is None


def test_run_debate_task_wraps_correctly(monkeypatch):
    fake_llm = _make_fake_llm(critic_agrees_after=0)
    monkeypatch.setattr(llm_client, "generate_structured", fake_llm, raising=False)

    async_result = tasks.run_debate.apply(args=["AAPL", "1W", date(2025, 12, 30)])

    assert async_result.successful()
    payload = async_result.result
    assert payload["direction"] == "BUY"

    status = debate_cache.get_status(async_result.id)
    assert status.status == "DONE"


def test_run_debate_task_accepts_iso_string_date(monkeypatch):
    """The API sends as_of_date as an ISO string (JSON task serializer)."""
    seen = {}

    def fake_run(ticker, horizon, as_of_date, task_id):
        seen["as_of_date"] = as_of_date

        class _R:
            def model_dump(self, mode):
                return {}

        return _R()

    monkeypatch.setattr(tasks, "_run_debate", fake_run)

    tasks.run_debate.apply(args=["AAPL", "1W", "2025-12-30"])

    assert seen["as_of_date"] == date(2025, 12, 30)
