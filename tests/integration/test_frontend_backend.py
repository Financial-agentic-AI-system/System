"""The frontend's REST client against the real backend, end to end.

Real parts: Postgres (prices loaded by the datalake loader), Redis (Celery
broker, result backend and debate cache), a Celery worker running the
LangGraph debate, the FastAPI app served by uvicorn over HTTP, and
`frontend/api_client.py` parsing every response with the frontend's own
models. The only stand-in is the LLM, which answers from a script
(`fake_llm`), so the run is deterministic and costs nothing.

Each response is also fetched raw and compared with what the frontend model
makes of it: a field the frontend drops, renames or re-types fails the test.
"""

import socket
import threading
import time
from datetime import date

import pytest
import requests
import uvicorn
from api_client import BackendClient
from celery.contrib.testing.worker import start_worker
from debate_model import build_rounds, describe_progress
from models import ApiError, DebateStatus, HistoryEntry, Meta, PredictResponse

from src.agents.state import MAX_ROUNDS
from src.api.v1.main import app as api_app
from src.worker import tasks  # noqa: F401  (registers run_debate on the app)
from src.worker.celery_app import app as celery_app
from tests.integration.conftest import TEST_REDIS_URL, trading_days

AS_OF = date(2025, 12, 30)
PRICE_DAYS = trading_days(date(2024, 6, 3), 400)  # 2024-06-03 .. 2025-12-12
# alternating +1% / -1%: a non-zero volatility for the PM's HOLD band
PRICES = [(day, 100.0 * (1.01 if i % 2 else 1.0)) for i, day in enumerate(PRICE_DAYS)]


@pytest.fixture
def worker(redis_db):
    """A real Celery worker thread on the test Redis."""
    celery_app.conf.update(broker_url=TEST_REDIS_URL, result_backend=TEST_REDIS_URL)
    with start_worker(celery_app, pool="solo", perform_ping_check=False):
        yield


@pytest.fixture
def backend_url(redis_db, db_session):
    """The FastAPI app served by uvicorn on a free local port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(api_app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            pytest.fail("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _raw(url: str) -> dict:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _assert_lossless(model: type, raw: dict) -> None:
    """The frontend model keeps every field of the backend's JSON, unchanged."""
    assert model.model_validate(raw).model_dump(mode="json") == raw


def _wait_until_finished(client: BackendClient, task_id: str) -> DebateStatus:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = client.status(task_id)
        if status.status in ("DONE", "FAILED"):
            return status
        time.sleep(0.2)
    pytest.fail(f"debate {task_id} did not finish: last status {status.status}")


def test_debate_started_from_the_frontend_runs_and_reads_back(
    backend_url, worker, load_prices, fake_llm, monkeypatch
):
    monkeypatch.setenv("SENTIMENT_RETRIEVAL", "0")  # no embedding API
    load_prices({"AAPL": PRICES})
    # the Critic disagrees once: round 2 re-asks only the Macro agent
    fake_llm(critic_agrees_after=1, selected_agents=["macro"])
    client = BackendClient(backend_url)
    api = f"{backend_url}/api/v1"

    # the forms: tickers and the backtest window come from the database
    assert client.health() is True
    assert client.tickers() == ["AAPL"]
    meta = client.meta()
    assert (meta.backtest_start, meta.backtest_end) == (
        date(2025, 1, 1),
        PRICE_DAYS[-1],
    )
    _assert_lossless(Meta, _raw(f"{api}/meta"))

    task_id = client.start("aapl", "1W", AS_OF)
    status = _wait_until_finished(client, task_id)

    assert status.status == "DONE", status.error
    assert (status.round_number, status.max_rounds) == (2, MAX_ROUNDS)
    _assert_lossless(DebateStatus, _raw(f"{api}/debate_status/{task_id}"))

    history = client.history(task_id)
    for entry in _raw(f"{api}/history/{task_id}")["history"]:
        _assert_lossless(HistoryEntry, entry)
    rounds = build_rounds(history)
    assert [r.number for r in rounds] == [1, 2]
    assert rounds[0].reasked == ["financial", "sentiment", "macro"]
    assert rounds[1].reasked == ["macro"]
    assert (rounds[0].critic.agree, rounds[1].critic.agree) == (False, True)
    assert describe_progress(rounds, status) == "Debate finished."

    response = client.result(task_id)
    _assert_lossless(PredictResponse, _raw(f"{api}/predict/result/{task_id}"))
    assert (response.ticker, response.horizon, response.as_of_date) == (
        "AAPL",
        "1W",
        AS_OF,
    )
    assert response.result.direction == "BUY"
    assert response.result.rounds_used == 2
    assert response.result.critic_agreed is True
    assert response.result.agent_reports.macro.opinion == "op"
    assert response.finished_at is not None


def test_backend_errors_reach_the_frontend_as_api_errors(backend_url, load_prices):
    load_prices({"AAPL": PRICES})
    client = BackendClient(backend_url)

    with pytest.raises(ApiError, match="Unknown ticker 'ZZZ'") as unknown_ticker:
        client.start("ZZZ", "1W", AS_OF)
    assert unknown_ticker.value.status_code == 422

    with pytest.raises(ApiError, match="as_of_date cannot be in the future"):
        client.start("AAPL", "1W", date(2999, 1, 1))

    with pytest.raises(ApiError) as unknown_task:
        client.status("no-such-task")
    assert unknown_task.value.status_code == 404
