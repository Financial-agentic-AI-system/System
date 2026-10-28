"""Unit tests for LLM usage logging (src/agents/usage.py) — no network."""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from pydantic import BaseModel

from src.agents import llm_client, usage


def test_cost_uses_per_million_prices(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_INPUT_PER_M", 0.25)
    monkeypatch.setattr(usage, "PRICE_OUTPUT_PER_M", 0.70)
    assert usage.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.95)
    assert usage.cost_usd(2_000, 500) == pytest.approx(0.00085)


def test_track_sums_calls_from_threads():
    # LangGraph runs parallel nodes in worker threads with a copy of the
    # context; the executor below does the same.
    call = ("x", {"prompt_tokens": 100, "completion_tokens": 10}, 0.5, 1)
    with usage.track() as debate:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for _ in range(3):
                pool.submit(contextvars.copy_context().run, usage.record_call, *call)
        usage.record_failure("y", 1.0, 5, "boom")

    assert (debate.calls, debate.failed_calls) == (3, 1)
    assert (debate.input_tokens, debate.output_tokens) == (300, 30)
    assert debate.llm_seconds == pytest.approx(2.5)
    assert "3 LLM call(s) (1 failed), 300 in / 30 out tokens" in debate.summary()


def test_record_outside_track_only_logs(caplog):
    caplog.set_level(logging.INFO, logger="src.agents.usage")
    usage.record_call("generate_structured", {"prompt_tokens": 7}, 0.1, 1)
    assert "debate=- node=- generate_structured: 7 in / 0 out tokens" in caplog.text


class _Answer(BaseModel):
    ok: bool


class _Response:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_llm_call_is_recorded(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.agents.usage")
    monkeypatch.setattr(llm_client, "_endpoint_url", lambda: "https://x")
    monkeypatch.setattr(llm_client, "_auth_headers", lambda: {})
    payload = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 80},
    }
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _Response(payload))

    with usage.track() as debate:
        assert llm_client.generate_structured("p", _Answer) == _Answer(ok=True)

    assert (debate.calls, debate.input_tokens, debate.output_tokens) == (1, 1200, 80)
    assert "schema '_Answer': 1200 in / 80 out tokens" in caplog.text


def test_failed_llm_call_is_recorded(monkeypatch, caplog):
    monkeypatch.setattr(llm_client, "_endpoint_url", lambda: "https://x")
    monkeypatch.setattr(llm_client, "_auth_headers", lambda: {})

    class _NotFound(_Response):
        status_code = 404

        def raise_for_status(self):
            raise requests.HTTPError("404", response=self)

    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _NotFound({}))

    with usage.track() as debate, pytest.raises(llm_client.LLMClientError):
        llm_client.generate_structured("p", _Answer)

    assert (debate.calls, debate.failed_calls) == (0, 1)
    assert "FAILED after 1 attempt(s)" in caplog.text


def test_lines_carry_the_debate_id(caplog):
    caplog.set_level(logging.INFO, logger="src.agents.usage")
    with usage.track(debate_id="abc123"):
        usage.record_call("generate_structured", {"prompt_tokens": 1}, 0.1, 1)
    assert "llm call debate=abc123 node=- generate_structured" in caplog.text


def test_retries_are_logged_and_summed(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.agents.usage")
    monkeypatch.setattr(llm_client, "_endpoint_url", lambda: "https://x")
    monkeypatch.setattr(llm_client, "_auth_headers", lambda: {})
    monkeypatch.setattr(llm_client, "_retry_delay", lambda exc, attempt: 2.0)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)

    class _RateLimited(_Response):
        status_code = 429

        def raise_for_status(self):
            raise requests.HTTPError("429", response=self)

    ok = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }
    replies = [_RateLimited({}), _RateLimited({}), _Response(ok)]
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: replies.pop(0))

    with usage.track(debate_id="d1") as debate:
        llm_client.generate_structured("p", _Answer)

    assert (
        "debate=d1 node=- generate_structured for schema '_Answer': "
        "HTTP 429, retry 2/5 in 2.0 s" in caplog.text
    )
    assert "retry 3/5" in caplog.text
    assert (debate.calls, debate.retries, debate.retry_wait_seconds) == (1, 2, 4.0)
    assert "2 retries (4.0 s waiting)" in debate.summary()
    assert ", 3 attempts" in caplog.text
