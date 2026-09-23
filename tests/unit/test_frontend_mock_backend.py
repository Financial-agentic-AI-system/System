"""Unit tests for frontend/mock_backend.py — the fake backend must honour the
same contract as the real worker/API (docs/ARCHITECTURE.md §4-5)."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from debate_model import build_rounds  # noqa: E402
from mock_backend import MockClient, build_script  # noqa: E402
from models import ApiError  # noqa: E402

T0 = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def test_script_is_deterministic():
    a, res_a = build_script("AAPL", "1W", date(2026, 1, 15))
    b, res_b = build_script("AAPL", "1W", date(2026, 1, 15))
    assert res_a == res_b
    assert [(e.node, e.round_number) for e in a] == [
        (e.node, e.round_number) for e in b
    ]


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "NVDA", "JPM", "GS", "TSLA"])
def test_script_respects_protocol(ticker):
    events, result = build_script(ticker, "1M", date(2026, 3, 2))
    times = [e.at for e in events]
    assert times == sorted(times)
    assert events[-1].node == "finalize"

    rounds = build_rounds(_to_history(events))
    assert 1 <= len(rounds) <= 3
    assert result["rounds_used"] == len(rounds)
    for r in rounds[:-1]:
        assert r.critic is not None and r.critic.agree is False
    last = rounds[-1]
    assert last.critic.agree == result["critic_agreed"]
    if len(rounds) < 3:
        assert result["critic_agreed"] is True  # only the round limit ends a dispute
    assert result["direction"] in {"BUY", "HOLD", "SELL"}
    assert 0.0 <= result["confidence"] <= 1.0


def _to_history(events):
    from models import HistoryEntry

    return [
        HistoryEntry(node=e.node, round_number=e.round_number, delta=e.delta)
        for e in events
    ]


def test_client_replays_debate_over_time():
    clock = Clock()
    client = MockClient(clock=clock)
    task_id = client.start("AAPL", "1W", date(2026, 1, 15))

    assert client.status(task_id).status == "PENDING"
    assert client.history(task_id) == []
    assert client.result(task_id).result is None

    clock.advance(1000)
    assert client.status(task_id).status == "DONE"
    resp = client.result(task_id)
    assert resp.status == "DONE"
    assert resp.result is not None
    assert resp.ticker == "AAPL" and resp.as_of_date == date(2026, 1, 15)
    assert resp.finished_at > resp.created_at
    assert client.history(task_id)[-1].node == "finalize"


def test_client_reports_running_midway():
    clock = Clock()
    client = MockClient(clock=clock)
    task_id = client.start("MSFT", "1W", date(2026, 1, 15))
    events, _ = build_script("MSFT", "1W", date(2026, 1, 15))

    clock.advance(events[1].at + 0.01)
    status = client.status(task_id)
    assert status.status == "RUNNING"
    assert status.current_node == events[1].node
    assert len(client.history(task_id)) == 2


def test_unknown_task_raises():
    with pytest.raises(ApiError):
        MockClient().status("nope")
