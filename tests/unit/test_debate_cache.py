"""Unit tests for src/cache/debate_cache.py — no real Redis, fakeredis only.

debate_cache.py imports `redis_client` by name, so patch `debate_cache.redis_client`
directly, not `redis_client_module.redis_client` (see module docstring there).
"""

import fakeredis
import pytest

from src.agents.state import AgentReport, AgentReports, PredictResult
from src.cache import debate_cache


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(debate_cache, "redis_client", fake)
    return fake


def test_get_status_returns_none_when_missing():
    assert debate_cache.get_status("nope") is None


def test_set_status_then_get_status_roundtrip():
    debate_cache.set_status("t1", "RUNNING", round_number=2, current_node="macro")
    status = debate_cache.get_status("t1")
    assert status.status == "RUNNING"
    assert status.round_number == 2
    assert status.current_node == "macro"
    assert status.error is None


def test_set_status_default_current_node_is_none():
    debate_cache.set_status("t1", "RUNNING", round_number=1)
    assert debate_cache.get_status("t1").current_node is None


def test_set_status_overwrites_previous_value():
    debate_cache.set_status("t1", "RUNNING", round_number=1, current_node="financial")
    debate_cache.set_status("t1", "DONE", round_number=1, current_node="finalize")
    assert debate_cache.get_status("t1").status == "DONE"


def test_set_status_with_error():
    debate_cache.set_status("t1", "FAILED", round_number=1, error="boom")
    status = debate_cache.get_status("t1")
    assert status.status == "FAILED"
    assert status.error == "boom"


def test_set_status_sets_ttl():
    debate_cache.set_status("t1", "RUNNING", round_number=1)
    ttl = debate_cache.redis_client.ttl(debate_cache._status_key("t1"))
    assert 0 < ttl <= debate_cache.CACHE_TTL_SECONDS


def test_append_history_accumulates_in_order():
    debate_cache.append_history(
        "t1",
        "financial",
        1,
        {"financial_report": AgentReport(opinion="o1", reasoning="r1")},
    )
    debate_cache.append_history(
        "t1",
        "sentiment",
        1,
        {"sentiment_report": AgentReport(opinion="o2", reasoning="r2")},
    )
    history = debate_cache.get_history("t1")
    assert [h.node for h in history] == ["financial", "sentiment"]
    assert history[0].delta["financial_report"]["opinion"] == "o1"


def test_append_history_keeps_both_entries_when_same_node_reports_twice():
    """macro reports in round 1, Critic disagrees, PM re-asks macro alone in
    round 2 with a different opinion — history must keep both, not
    overwrite the first with the second.
    """
    debate_cache.append_history(
        "t1", "macro", 1, {"macro_report": AgentReport(opinion="BUY", reasoning="r1")}
    )
    debate_cache.append_history(
        "t1",
        "macro",
        2,
        {"macro_report": AgentReport(opinion="HOLD", reasoning="r2")},
    )
    history = debate_cache.get_history("t1")
    assert len(history) == 2
    assert [h.round_number for h in history] == [1, 2]
    assert history[0].delta["macro_report"]["opinion"] == "BUY"
    assert history[1].delta["macro_report"]["opinion"] == "HOLD"


def test_get_history_empty_when_missing():
    assert debate_cache.get_history("nope") == []


def test_set_result_then_get_result_roundtrip():
    result = PredictResult(
        direction="BUY",
        confidence=0.7,
        summary="s",
        rounds_used=2,
        critic_agreed=True,
        agent_reports=AgentReports(
            financial=AgentReport(opinion="o", reasoning="r"),
            sentiment=AgentReport(opinion="o", reasoning="r"),
            macro=AgentReport(opinion="o", reasoning="r"),
        ),
    )
    debate_cache.set_result("t1", result)
    fetched = debate_cache.get_result("t1")
    assert fetched.direction == "BUY"
    assert fetched.confidence == 0.7


def test_get_result_returns_none_when_missing():
    assert debate_cache.get_result("nope") is None
