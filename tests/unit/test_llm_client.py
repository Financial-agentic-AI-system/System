"""Unit tests for src/agents/llm_client.py — no network beyond localhost.

The retry tests talk to a local HTTP server that answers from a script (an
error status, a broken body, a dropped connection), so `requests` goes
through a real socket exactly as it does against Vertex AI. Only the endpoint
URL, the auth headers and `time.sleep` are replaced.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from src.agents import llm_client


class Selection(BaseModel):
    agents: list[str]


def _reply(answer: object) -> dict:
    return {"choices": [{"message": {"content": json.dumps(answer)}}]}


# --------------------------------------------------------------------------- #
# Parsing structured answers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "answer",
    [
        {"agents": ["financial"]},
        # Llama 4 wrapping the answer as a function call
        {
            "type": "function",
            "name": "Selection",
            "parameters": {"agents": ["financial"]},
        },
        {"type": "function", "name": "x", "arguments": '{"agents": ["financial"]}'},
    ],
)
def test_generate_structured_accepts_plain_and_wrapped_answers(monkeypatch, answer):
    monkeypatch.setattr(
        llm_client, "_call", lambda body, parse, *a, **k: parse(_reply(answer))
    )

    assert llm_client.generate_structured("p", Selection) == Selection(
        agents=["financial"]
    )


def test_unrelated_type_field_is_left_alone():
    answer = {"type": "BUY", "agents": []}
    assert llm_client._unwrap_function_call(answer) is answer


# --------------------------------------------------------------------------- #
# HTTP: a scripted local server
# --------------------------------------------------------------------------- #
DROP = "drop"  # close the connection without answering


def _ok(answer: object = None, **message) -> tuple:
    if answer is not None:
        message["content"] = json.dumps(answer)
    return 200, {}, {"choices": [{"message": message}]}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.server.requests.append(json.loads(self.rfile.read(length)))
        step = self.server.script.pop(0)
        if step == DROP:
            self.close_connection = True
            return
        status, headers, body = step
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    """Returns (script, requests, sleeps): queue responses into `script`; the
    request bodies the client sent and the delays it waited are recorded."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.script, httpd.requests = [], []
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()

    url = f"http://127.0.0.1:{httpd.server_address[1]}/chat/completions"
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(llm_client, "_endpoint_url", lambda: url)
    monkeypatch.setattr(llm_client, "_auth_headers", lambda: {})
    sleeps: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 0.5)

    yield httpd.script, httpd.requests, sleeps
    httpd.shutdown()
    httpd.server_close()


def test_request_carries_the_schema_and_no_tools(server):
    script, requests, _ = server
    script.append(_ok({"agents": ["macro"]}))

    assert llm_client.generate_structured("prompt", Selection) == Selection(
        agents=["macro"]
    )
    [body] = requests
    assert body["messages"] == [{"role": "user", "content": "prompt"}]
    assert body["temperature"] == 0.0
    assert body["response_format"]["json_schema"]["name"] == "Selection"
    # no grounding / search tool on structured calls (docs/evaluation.md §2)
    assert "tools" not in body


def test_transient_failures_are_retried_until_an_answer_parses(server):
    script, requests, sleeps = server
    script += [
        (503, {}, {"error": "unavailable"}),  # server error
        DROP,  # connection dropped, no response at all
        (200, {}, b"<html>proxy error</html>"),  # body is not JSON
        _ok({"wrong": "shape"}),  # JSON, but not the schema
        _ok({"agents": ["financial"]}),
    ]

    assert llm_client.generate_structured("p", Selection) == Selection(
        agents=["financial"]
    )
    assert len(requests) == 5
    assert sleeps == [2, 4, 8, 16]  # 2 ** attempt


def test_rate_limit_honours_retry_after_then_backs_off_harder(server):
    script, _, sleeps = server
    script += [
        (429, {"Retry-After": "7"}, {}),
        (429, {}, {}),
        _ok({"agents": []}),
    ]

    llm_client.generate_structured("p", Selection)

    # no header on the 2nd attempt: min(60, 5 * 2**2) + jitter (0.5 here)
    assert sleeps == [7.0, 20.5]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_errors_fail_at_once(server, status):
    script, requests, sleeps = server
    script.append((status, {}, {"error": "no"}))

    with pytest.raises(llm_client.LLMClientError, match="after 1 attempt"):
        llm_client.generate_structured("p", Selection)
    assert (len(requests), sleeps) == (1, [])


def test_gives_up_after_the_last_attempt(server):
    script, requests, sleeps = server
    script += [(503, {}, {})] * llm_client._MAX_RETRIES

    with pytest.raises(llm_client.LLMClientError, match="after 5 attempt"):
        llm_client.generate_structured("p", Selection)
    assert len(requests) == 5
    assert len(sleeps) == 4  # no wait after the last attempt


def test_tool_calls_are_parsed_and_malformed_ones_skipped(server):
    script, requests, _ = server
    tool_calls = [
        {"function": {"name": "search_articles", "arguments": '{"queries": ["a"]}'}},
        {"function": {"name": "search_articles", "arguments": "{not json"}},
        {"function": {"arguments": "{}"}},  # no name
    ]
    script += [_ok(content=None, tool_calls=tool_calls), _ok(content="plain text")]
    tools = [{"type": "function", "function": {"name": "search_articles"}}]

    [call] = llm_client.generate_tool_calls("p", tools)
    assert (call.name, call.arguments) == ("search_articles", {"queries": ["a"]})
    assert requests[0]["tools"] == tools
    assert "response_format" not in requests[0]

    # a plain-text answer means the model asked for nothing
    assert llm_client.generate_tool_calls("p", tools) == []
