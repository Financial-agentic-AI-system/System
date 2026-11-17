"""The backtest proxy's allowlist (docker/backtest-proxy/squid.conf).

During a backtest the worker reaches the internet only through this proxy, so
the allowlist is what keeps the debate from seeing today's prices or news
(docs/evaluation.md §1, channel 1). Each test asks the proxy to open a
connection to one host and checks whether the proxy lets it through.

The tests run against squid started with the repository's own config; they
are skipped when it is not running (and fail on CI):

    docker run -d --name backtest-proxy -p 8888:8888 \\
        -v "$PWD/docker/backtest-proxy/squid.conf:/etc/squid/squid.conf:ro" \\
        ubuntu/squid:latest

    TEST_PROXY_URL  http://localhost:8888 (default)

A refusal by the allowlist is squid's 403 with `ERR_ACCESS_DENIED`. A host that
is let through but cannot be reached (no internet on a dev machine) gets a
different error, so only on CI, which has internet, must the connection open.
"""

import os
import socket
from urllib.parse import urlsplit

import pytest

from tests.integration.conftest import _unavailable

PROXY = urlsplit(os.getenv("TEST_PROXY_URL", "http://localhost:8888"))
DENIED = "ERR_ACCESS_DENIED"

# The hosts the backtest needs: LLM inference and the ADC token refresh.
ALLOWED = [
    "us-east5-aiplatform.googleapis.com",
    "oauth2.googleapis.com",
]

# Hosts the debate must not reach, and why.
REFUSED = [
    ("www.google.com", "Google Search: today's news"),
    ("generativelanguage.googleapis.com", "Gemini API, grounding with search"),
    ("aiplatform.googleapis.com", "the global endpoint, not the allowed region"),
    ("us-central1-aiplatform.googleapis.com", "another region"),
    ("storage.googleapis.com", "another Google product on googleapis.com"),
    ("www.alphavantage.co", "fundamentals and news"),
    ("query1.finance.yahoo.com", "prices"),
    ("api.stlouisfed.org", "FRED macro series"),
    ("oauth2.googleapis.com.example.com", "a look-alike of an allowed host"),
    ("example.com", "any other site"),
]


@pytest.fixture(scope="module")
def proxy():
    try:
        socket.create_connection((PROXY.hostname, PROXY.port), timeout=3).close()
    except OSError as exc:
        _unavailable(f"backtest proxy not reachable at {PROXY.geturl()}: {exc}")
    return PROXY.hostname, PROXY.port


def _ask(proxy, request: str) -> tuple[int, str | None]:
    """Send one raw request to the proxy; return its status and squid's error."""
    with socket.create_connection(proxy, timeout=20) as sock:
        sock.sendall(request.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                break
            head += chunk
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    error = next(
        (
            line.split(":", 1)[1].split()[0]
            for line in lines[1:]
            if line.lower().startswith("x-squid-error:")
        ),
        None,
    )
    return status, error


def _connect(proxy, host: str, port: int = 443) -> tuple[int, str | None]:
    """An HTTPS tunnel, the way `requests` goes through HTTPS_PROXY."""
    target = f"{host}:{port}"
    return _ask(proxy, f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n")


@pytest.mark.parametrize("host", ALLOWED)
def test_allowed_hosts_are_let_through(proxy, host):
    status, error = _connect(proxy, host)
    assert error != DENIED
    if os.getenv("CI"):
        assert status == 200, f"tunnel to {host} did not open: {status} {error}"


@pytest.mark.parametrize(("host", "why"), REFUSED, ids=[h for h, _ in REFUSED])
def test_other_hosts_are_refused(proxy, host, why):
    assert _connect(proxy, host) == (403, DENIED), why


def test_allowed_host_on_another_port_is_refused(proxy):
    assert _connect(proxy, "oauth2.googleapis.com", 8443) == (403, DENIED)


def test_plain_http_to_another_site_is_refused(proxy):
    request = "GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n"
    assert _ask(proxy, request) == (403, DENIED)
