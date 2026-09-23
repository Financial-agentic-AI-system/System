"""Unit tests for src/api/v1 — the 4 REST endpoints of docs/ARCHITECTURE.md §4.

No real Redis or broker: `debate_cache.redis_client` is a fakeredis instance
and `celery_app.send_task` is replaced by a recorder. The cache is seeded
through the same `debate_cache` functions the worker uses, so these tests
also pin down that the API reads exactly what the worker writes.
"""

from datetime import date, datetime, timedelta, timezone

import fakeredis
import pytest
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError

from src.agents.state import AgentReport, AgentReports, PredictResult
from src.api.v1.endpoints import predict
from src.api.v1.main import app
from src.cache import debate_cache
from src.cache.schemas import TaskMeta

AS_OF = "2025-12-30"


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(debate_cache, "redis_client", fake)
    return fake


@pytest.fixture
def sent(monkeypatch):
    """Records every send_task call instead of talking to a broker."""
    calls: list[dict] = []

    def fake_send_task(name, args=None, task_id=None, **kwargs):
        calls.append({"name": name, "args": args, "task_id": task_id})

    monkeypatch.setattr(predict.celery_app, "send_task", fake_send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(app)


def _result() -> PredictResult:
    report = AgentReport(opinion="o", reasoning="r")
    return PredictResult(
        direction="BUY",
        confidence=0.7,
        summary="s",
        rounds_used=2,
        critic_agreed=True,
        agent_reports=AgentReports(financial=report, sentiment=report, macro=report),
    )


def _seed(task_id: str = "t1") -> None:
    debate_cache.set_meta(
        task_id,
        TaskMeta(
            ticker="AAPL",
            horizon="1W",
            as_of_date=date(2025, 12, 30),
            model_version="test-model",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )


# --------------------------------------------------------------------------- #
# POST /predict/start
# --------------------------------------------------------------------------- #
def test_start_enqueues_task_and_returns_task_id(client, sent):
    resp = client.post(
        "/api/v1/predict/start",
        json={"ticker": " aapl ", "horizon": "1W", "as_of_date": AS_OF},
    )

    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    assert resp.json()["status"] == "PENDING"

    # ticker normalized, date sent as an ISO string, celery id == our task_id
    assert sent == [
        {
            "name": "src.worker.tasks.run_debate",
            "args": ["AAPL", "1W", AS_OF],
            "task_id": task_id,
        }
    ]
    assert debate_cache.get_status(task_id).status == "PENDING"
    meta = debate_cache.get_meta(task_id)
    assert (meta.ticker, meta.horizon, meta.as_of_date) == (
        "AAPL",
        "1W",
        date(2025, 12, 30),
    )


@pytest.mark.parametrize(
    "body",
    [
        {"ticker": "AAPL", "horizon": "2Y", "as_of_date": AS_OF},  # bad horizon
        {"ticker": "", "horizon": "1W", "as_of_date": AS_OF},
        {"ticker": "AA PL", "horizon": "1W", "as_of_date": AS_OF},
        {"ticker": "AAPL", "horizon": "1W", "as_of_date": "not-a-date"},
        {"ticker": "AAPL", "horizon": "1W"},  # as_of_date is required
        {
            "ticker": "AAPL",
            "horizon": "1W",
            "as_of_date": (date.today() + timedelta(days=1)).isoformat(),
        },
    ],
)
def test_start_rejects_invalid_body(client, sent, body):
    resp = client.post("/api/v1/predict/start", json=body)
    assert resp.status_code == 422
    assert sent == []


def test_start_returns_503_and_marks_failed_when_broker_is_down(
    monkeypatch, sent, client
):
    def boom(*args, **kwargs):
        raise OperationalError("broker down")

    monkeypatch.setattr(predict.celery_app, "send_task", boom)
    resp = client.post(
        "/api/v1/predict/start",
        json={"ticker": "AAPL", "horizon": "1W", "as_of_date": AS_OF},
    )

    assert resp.status_code == 503
    # the accepted-but-never-enqueued task must not sit at PENDING forever
    [key] = [k for k in debate_cache.redis_client.keys("*:status")]
    task_id = key.split(":")[1]
    status = debate_cache.get_status(task_id)
    assert status.status == "FAILED"
    assert "broker down" in status.error


# --------------------------------------------------------------------------- #
# GET /debate_status/{task_id}
# --------------------------------------------------------------------------- #
def test_debate_status_unknown_task_is_404(client):
    assert client.get("/api/v1/debate_status/nope").status_code == 404


def test_debate_status_returns_cached_status(client):
    debate_cache.set_status("t1", "RUNNING", round_number=2, current_node="macro")

    body = client.get("/api/v1/debate_status/t1").json()

    assert body["status"] == "RUNNING"
    assert body["round_number"] == 2
    assert body["current_node"] == "macro"
    assert body["error"] is None


# --------------------------------------------------------------------------- #
# GET /history/{task_id}
# --------------------------------------------------------------------------- #
def test_history_unknown_task_is_404(client):
    assert client.get("/api/v1/history/nope").status_code == 404


def test_history_empty_for_queued_task(client):
    debate_cache.set_status("t1", "PENDING", round_number=1)
    assert client.get("/api/v1/history/t1").json() == {"task_id": "t1", "history": []}


def test_history_returns_entries_in_order(client):
    debate_cache.set_status("t1", "RUNNING", round_number=2)
    debate_cache.append_history(
        "t1",
        "macro",
        1,
        {"macro_report": AgentReport(opinion="o1", reasoning="r1")},
    )
    debate_cache.append_history(
        "t1", "pm_select_next_agents", 2, {"active_agents": ["macro"]}
    )

    body = client.get("/api/v1/history/t1").json()

    assert [h["node"] for h in body["history"]] == ["macro", "pm_select_next_agents"]
    assert [h["round_number"] for h in body["history"]] == [1, 2]
    assert body["history"][0]["delta"]["macro_report"]["opinion"] == "o1"
    assert body["history"][1]["delta"]["active_agents"] == ["macro"]


# --------------------------------------------------------------------------- #
# GET /predict/result/{task_id}
# --------------------------------------------------------------------------- #
def test_result_unknown_task_is_404(client):
    assert client.get("/api/v1/predict/result/nope").status_code == 404


def test_result_without_meta_is_404(client):
    """A task the API never accepted (no request metadata) has no envelope."""
    debate_cache.set_status("t1", "RUNNING", round_number=1)
    assert client.get("/api/v1/predict/result/t1").status_code == 404


def test_result_while_running_has_no_result(client):
    _seed()
    debate_cache.set_status("t1", "RUNNING", round_number=1, current_node="macro")

    body = client.get("/api/v1/predict/result/t1").json()

    assert body["status"] == "RUNNING"
    assert body["result"] is None
    assert body["finished_at"] is None
    assert (body["ticker"], body["horizon"], body["as_of_date"]) == (
        "AAPL",
        "1W",
        "2025-12-30",
    )


def test_result_done_returns_full_envelope(client):
    _seed()
    debate_cache.set_result("t1", _result())
    debate_cache.set_status("t1", "DONE", round_number=2, current_node="finalize")

    body = client.get("/api/v1/predict/result/t1").json()

    assert body["status"] == "DONE"
    assert body["model_version"] == "test-model"
    assert body["result"]["direction"] == "BUY"
    assert body["result"]["rounds_used"] == 2
    assert body["result"]["agent_reports"]["macro"] == {
        "opinion": "o",
        "reasoning": "r",
    }
    assert body["finished_at"] is not None
    assert body["error"] is None


def test_result_done_but_result_expired_is_404(client):
    _seed()
    debate_cache.set_status("t1", "DONE", round_number=1, current_node="finalize")
    assert client.get("/api/v1/predict/result/t1").status_code == 404


def test_result_failed_carries_error_and_no_result(client):
    _seed()
    debate_cache.set_status("t1", "FAILED", round_number=1, error="LLM exploded")

    body = client.get("/api/v1/predict/result/t1").json()

    assert body["status"] == "FAILED"
    assert body["result"] is None
    assert body["error"] == "LLM exploded"
    assert body["finished_at"] is not None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
