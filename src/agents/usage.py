"""Technical logging of LLM calls: tokens, cost and time, per call and per
debate.

Every call through `llm_client` is logged on one INFO line. Inside `track()`
(the worker wraps each debate in it) the calls are also summed, so the end of
a debate can log its totals. The LangGraph node making the call is read from
the graph's run config, so the agent code does not have to pass it.

Prices are USD per 1M tokens, overridable for another model:
LLM_PRICE_INPUT_PER_M, LLM_PRICE_OUTPUT_PER_M (defaults: Llama 4 Scout MaaS).
"""

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PRICE_INPUT_PER_M = float(os.environ.get("LLM_PRICE_INPUT_PER_M", "0.25"))
PRICE_OUTPUT_PER_M = float(os.environ.get("LLM_PRICE_OUTPUT_PER_M", "0.70"))


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M
    ) / 1_000_000


@dataclass
class DebateUsage:
    debate_id: str | None = None
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    retries: int = 0
    retry_wait_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int, seconds: float) -> None:
        # The three agents call the model from parallel threads.
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.llm_seconds += seconds

    def add_failure(self, seconds: float) -> None:
        with self._lock:
            self.failed_calls += 1
            self.llm_seconds += seconds

    def add_retry(self, wait_seconds: float) -> None:
        with self._lock:
            self.retries += 1
            self.retry_wait_seconds += wait_seconds

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens)

    def summary(self) -> str:
        failed = f" ({self.failed_calls} failed)" if self.failed_calls else ""
        retries = (
            f", {self.retries} retries ({self.retry_wait_seconds:.1f} s waiting)"
            if self.retries
            else ""
        )
        return (
            f"{self.calls} LLM call(s){failed}, {self.input_tokens} in / "
            f"{self.output_tokens} out tokens, ${self.cost_usd:.4f}, "
            f"{self.llm_seconds:.1f} s in LLM calls{retries}"
        )


_current: ContextVar[DebateUsage | None] = ContextVar("debate_usage", default=None)


@contextmanager
def track(debate_id: str | None = None) -> Iterator[DebateUsage]:
    """Sum every LLM call made inside the block (including LangGraph's worker
    threads, which inherit the context) and tag its log lines with
    `debate_id`."""
    usage = DebateUsage(debate_id=debate_id)
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def current_node() -> str | None:
    """The LangGraph node running this call, or None outside a graph."""
    try:
        from langgraph.config import get_config

        return get_config()["metadata"].get("langgraph_node")
    except Exception:
        return None


def _where() -> str:
    """`debate=<id> node=<node>` prefix, so lines of parallel debates can be
    told apart."""
    tracker = _current.get()
    debate = tracker.debate_id if tracker and tracker.debate_id else "-"
    return f"debate={debate} node={current_node() or '-'}"


def _short_error(error: object) -> str:
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None):
        return f"HTTP {response.status_code}"
    return f"{type(error).__name__}: {str(error)[:80]}"


def record_retry(
    what: str, attempt: int, max_attempts: int, wait_seconds: float, error: object
) -> None:
    logger.warning(
        "llm call %s %s: %s, retry %d/%d in %.1f s",
        _where(),
        what,
        _short_error(error),
        attempt + 1,
        max_attempts,
        wait_seconds,
    )
    tracker = _current.get()
    if tracker is not None:
        tracker.add_retry(wait_seconds)


def record_call(what: str, usage: dict | None, seconds: float, attempts: int) -> None:
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    logger.info(
        "llm call %s %s: %d in / %d out tokens, $%.5f, %.2f s%s",
        _where(),
        what,
        input_tokens,
        output_tokens,
        cost_usd(input_tokens, output_tokens),
        seconds,
        f", {attempts} attempts" if attempts > 1 else "",
    )
    tracker = _current.get()
    if tracker is not None:
        tracker.add(input_tokens, output_tokens, seconds)


def record_failure(what: str, seconds: float, attempts: int, error: object) -> None:
    logger.warning(
        "llm call %s %s FAILED after %d attempt(s), %.2f s: %s",
        _where(),
        what,
        attempts,
        seconds,
        error,
    )
    tracker = _current.get()
    if tracker is not None:
        tracker.add_failure(seconds)
