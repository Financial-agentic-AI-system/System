"""Unit tests for frontend/api_client.py — HTTP is stubbed, no backend needed."""

import sys
from datetime import date
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

import api_client  # noqa: E402
from api_client import BackendClient  # noqa: E402
from models import ApiError  # noqa: E402

REPORT = {"opinion": "o", "reasoning": "r"}
RESULT = {
    "direction": "BUY",
    "confidence": 0.8,
    "summary": "s",
    "rounds_used": 1,
    "critic_agreed": True,
    "agent_reports": {"financial": REPORT, "sentiment": REPORT, "macro": REPORT},
}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Queue responses; records the calls made through `requests`."""
    calls: list[tuple[str, str, dict]] = []
    queue: list = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    monkeypatch.setattr(
        api_client.requests, "get", lambda url, **kw: fake_request("GET", url, **kw)
    )
    return queue, calls


def _client() -> BackendClient:
    return BackendClient("http://api:8000")


def test_start_posts_contract_body(http):
    queue, calls = http
    queue.append(FakeResponse(202, {"task_id": "abc", "status": "PENDING"}))

    assert _client().start("TSLA", "1W", date(2026, 1, 15)) == "abc"
    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", "http://api:8000/api/v1/predict/start")
    assert kwargs["json"] == {
        "ticker": "TSLA",
        "horizon": "1W",
        "as_of_date": "2026-01-15",
    }


def test_status_history_result_are_parsed(http):
    queue, _ = http
    queue += [
        FakeResponse(
            payload={
                "status": "RUNNING",
                "round_number": 2,
                "current_node": "critic",
                "updated_at": "2026-01-15T10:00:00Z",
                "error": None,
            }
        ),
        FakeResponse(
            payload={
                "task_id": "abc",
                "history": [
                    {
                        "node": "financial",
                        "round_number": 1,
                        "ts": "2026-01-15T10:00:00Z",
                        "delta": {"financial_report": REPORT},
                    }
                ],
            }
        ),
        FakeResponse(
            payload={
                "task_id": "abc",
                "ticker": "TSLA",
                "horizon": "1W",
                "status": "DONE",
                "as_of_date": "2026-01-15",
                "model_version": "m",
                "result": RESULT,
                "created_at": "2026-01-15T10:00:00Z",
                "finished_at": "2026-01-15T10:01:00Z",
                "error": None,
            }
        ),
    ]
    client = _client()

    status = client.status("abc")
    assert (status.status, status.round_number, status.current_node) == (
        "RUNNING",
        2,
        "critic",
    )
    history = client.history("abc")
    assert [e.node for e in history] == ["financial"]
    resp = client.result("abc")
    assert resp.result.direction == "BUY"
    assert resp.as_of_date == date(2026, 1, 15)


def test_unreachable_backend_raises_api_error(http):
    queue, _ = http
    queue.append(requests.ConnectionError("refused"))
    with pytest.raises(ApiError, match="Cannot reach the backend"):
        _client().status("abc")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (FakeResponse(404, {"detail": "Unknown task_id (kept 24 h)."}), "Unknown"),
        (
            FakeResponse(
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "as_of_date"],
                            "msg": "Value error, as_of_date cannot be in the future",
                        }
                    ]
                },
            ),
            "as_of_date: Value error, as_of_date cannot be in the future",
        ),
        (FakeResponse(503, {"detail": "Redis/broker unavailable: x"}), "Redis"),
        (FakeResponse(500, text="boom"), "500"),
    ],
)
def test_error_responses_become_readable_api_errors(http, response, expected):
    queue, _ = http
    queue.append(response)
    with pytest.raises(ApiError, match=expected):
        _client().start("TSLA", "1W", date(2026, 1, 15))


def test_unexpected_shape_raises_api_error(http):
    queue, _ = http
    queue.append(FakeResponse(payload={"status": "WHAT"}))
    with pytest.raises(ApiError, match="Unexpected response shape"):
        _client().status("abc")


def test_health(http):
    queue, calls = http
    queue += [FakeResponse(payload={"status": "ok"}), requests.ConnectionError("x")]
    client = _client()
    assert client.health() is True
    assert calls[0][1] == "http://api:8000/health"
    assert client.health() is False


def test_tickers(http):
    queue, calls = http
    queue.append(FakeResponse(payload={"tickers": ["AAPL", "TSLA"]}))

    assert _client().tickers() == ["AAPL", "TSLA"]
    assert calls[0][:2] == ("GET", "http://api:8000/api/v1/tickers")


def test_meta(http):
    queue, calls = http
    queue.append(
        FakeResponse(
            payload={
                "horizons": ["1W", "1M"],
                "backtest_start": "2025-01-01",
                "backtest_end": "2026-06-30",
                "max_rounds": 3,
            }
        )
    )

    meta = _client().meta()

    assert calls[0][:2] == ("GET", "http://api:8000/api/v1/meta")
    assert meta.horizons == ["1W", "1M"] and meta.max_rounds == 3
    assert meta.backtest_end == date(2026, 6, 30)


def test_api_error_carries_http_status(http):
    queue, _ = http
    queue += [FakeResponse(404, {"detail": "gone"}), requests.ConnectionError("x")]

    with pytest.raises(ApiError) as not_found:
        _client().status("abc")
    assert not_found.value.status_code == 404

    with pytest.raises(ApiError) as unreachable:
        _client().status("abc")
    assert unreachable.value.status_code is None
