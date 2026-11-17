"""Unit tests for src/worker/tasks.py — the Celery task that runs the debate
graph and mirrors its progress into src/cache/debate_cache.py.

Tests call `execute_debate` directly — no Celery machinery involved — except
the ones that go through the `@app.task` wrapper via `.apply()`, to prove the
wrapping itself works.
"""

import logging
from datetime import date

import pytest

from src.cache import debate_cache
from src.worker import tasks

pytestmark = pytest.mark.usefixtures("no_real_db", "fake_redis")

AS_OF = date(2025, 12, 30)


def test_run_debate_happy_path_writes_status_result_history_and_log(fake_llm, caplog):
    caplog.set_level(logging.INFO, logger="src.worker.tasks")
    fake_llm(critic_agrees_after=0)

    result = tasks.execute_debate("AAPL", "1W", AS_OF, task_id="t1")

    assert result.direction == "BUY"
    assert result.rounds_used == 1

    status = debate_cache.get_status("t1")
    assert status.status == "DONE"
    assert status.round_number == 1
    assert status.current_node == "finalize"

    assert debate_cache.get_result("t1").direction == "BUY"

    history = debate_cache.get_history("t1")
    # financial/sentiment/macro run in parallel — order among the three
    # isn't guaranteed, only that all three land before the sequential tail.
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert [h.node for h in history[3:]] == ["pm_synthesize", "critic", "finalize"]
    assert all(h.round_number == 1 for h in history)

    assert "debate t1 AAPL 1W as of 2025-12-30 started" in caplog.text
    assert "debate t1 AAPL 1W as of 2025-12-30 done: BUY, confidence 60%" in caplog.text
    assert "LLM call(s)" in caplog.text


def test_run_debate_partial_reask_tags_round_2_entries(fake_llm):
    fake_llm(critic_agrees_after=1, selected_agents=["macro"])

    result = tasks.execute_debate("AAPL", "1W", AS_OF, task_id="t1")

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

    assert debate_cache.get_status("t1").round_number == 2


def test_run_debate_marks_failed_on_exception_and_preserves_partial_history(fake_llm):
    fake_llm(critic_agrees_after=0, fail_on=("CriticFeedback", 1))

    with pytest.raises(RuntimeError):
        tasks.execute_debate("AAPL", "1W", AS_OF, task_id="t1")

    status = debate_cache.get_status("t1")
    assert status.status == "FAILED"
    assert status.error is not None

    # financial/sentiment/macro/pm_synthesize made it in before critic raised
    history = debate_cache.get_history("t1")
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert [h.node for h in history[3:]] == ["pm_synthesize"]

    assert debate_cache.get_result("t1") is None


def test_run_debate_task_wraps_correctly(fake_llm):
    fake_llm(critic_agrees_after=0)

    async_result = tasks.run_debate.apply(args=["AAPL", "1W", AS_OF])

    assert async_result.successful()
    assert async_result.result["direction"] == "BUY"
    assert debate_cache.get_status(async_result.id).status == "DONE"


def test_run_debate_task_accepts_iso_string_date(monkeypatch):
    """The API sends as_of_date as an ISO string (JSON task serializer)."""
    seen = {}

    def fake_run(ticker, horizon, as_of_date, task_id):
        seen["as_of_date"] = as_of_date

        class _R:
            def model_dump(self, mode):
                return {}

        return _R()

    monkeypatch.setattr(tasks, "execute_debate", fake_run)

    tasks.run_debate.apply(args=["AAPL", "1W", "2025-12-30"])

    assert seen["as_of_date"] == AS_OF
