"""Unit tests for src/cache/debate_cache.py — no real Redis, fakeredis only."""

from datetime import date, datetime, timezone

import pytest

from src.agents.state import AgentReport, AgentReports, PredictResult
from src.cache import debate_cache
from src.cache.schemas import TaskMeta

pytestmark = pytest.mark.usefixtures("fake_redis")


def _ttl(key: str) -> int:
    return debate_cache.redis_client.ttl(key)


def test_unknown_task_reads_as_empty():
    assert debate_cache.get_status("nope") is None
    assert debate_cache.get_history("nope") == []
    assert debate_cache.get_result("nope") is None
    assert debate_cache.get_meta("nope") is None


def test_set_status_roundtrip_overwrite_and_ttl():
    debate_cache.set_status("t1", "RUNNING", round_number=1)
    status = debate_cache.get_status("t1")
    assert (status.status, status.round_number) == ("RUNNING", 1)
    assert status.current_node is None  # the default
    assert status.error is None
    assert 0 < _ttl(debate_cache._status_key("t1")) <= debate_cache.CACHE_TTL_SECONDS

    debate_cache.set_status("t1", "RUNNING", round_number=2, current_node="macro")
    status = debate_cache.get_status("t1")
    assert (status.round_number, status.current_node) == (2, "macro")


def test_set_status_with_error():
    debate_cache.set_status("t1", "FAILED", round_number=1, error="boom")
    status = debate_cache.get_status("t1")
    assert status.status == "FAILED"
    assert status.error == "boom"


def test_append_history_keeps_every_entry_in_order():
    """macro reports in round 1, Critic disagrees, PM re-asks macro alone in
    round 2 with a different opinion — history must keep both, not
    overwrite the first with the second.
    """
    for node, round_number, opinion in [
        ("financial", 1, "o1"),
        ("macro", 1, "BUY"),
        ("macro", 2, "HOLD"),
    ]:
        debate_cache.append_history(
            "t1",
            node,
            round_number,
            {f"{node}_report": AgentReport(opinion=opinion, reasoning="r")},
        )

    history = debate_cache.get_history("t1")

    assert [(h.node, h.round_number) for h in history] == [
        ("financial", 1),
        ("macro", 1),
        ("macro", 2),
    ]
    assert history[0].delta["financial_report"]["opinion"] == "o1"
    assert history[1].delta["macro_report"]["opinion"] == "BUY"
    assert history[2].delta["macro_report"]["opinion"] == "HOLD"


def test_set_result_then_get_result_roundtrip():
    report = AgentReport(opinion="o", reasoning="r")
    result = PredictResult(
        direction="BUY",
        confidence=0.7,
        summary="s",
        rounds_used=2,
        critic_agreed=True,
        agent_reports=AgentReports(financial=report, sentiment=report, macro=report),
    )
    debate_cache.set_result("t1", result)
    fetched = debate_cache.get_result("t1")
    assert fetched.direction == "BUY"
    assert fetched.confidence == 0.7


def test_set_meta_then_get_meta_roundtrip():
    meta = TaskMeta(
        ticker="AAPL",
        horizon="1W",
        as_of_date=date(2025, 12, 30),
        model_version="m",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    debate_cache.set_meta("t1", meta)
    assert debate_cache.get_meta("t1") == meta
    assert 0 < _ttl(debate_cache._meta_key("t1")) <= debate_cache.CACHE_TTL_SECONDS
