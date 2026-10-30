"""LLM client for Gemini Enterprise Agent Platform — Llama 4 Scout 17B-16E
Instruct via MaaS. Cutoff (Aug 2024) and no-grounding rationale: docs/evaluation.md
§§2-3.3. Update that doc's "Model + version" row if this changes.

Two entry points: `generate_structured` (an answer in a Pydantic schema) and
`generate_tool_calls` (the model may ask for one of *our* local function tools).
Neither ever attaches a grounding/search tool.

Auth: ADC by default (`gcloud auth application-default login`). Set
GEMINI_API_KEY to use an API key instead — untested against this specific
endpoint, treat as a local-testing convenience, not the recommended path.

Env vars (see .env.example): GCP_PROJECT_ID (required), GCP_LOCATION
(default us-east5), LLM_MODEL_VERSION, GEMINI_API_KEY (optional).

Requires accepting the Llama Community License Agreement on the model's
Model Garden page once per project, or every call 404s.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import google.auth
import google.auth.transport.requests
import requests
from pydantic import BaseModel, ValidationError

from src.agents import usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")  # whatever a response parser returns

MODEL_VERSION = os.environ.get(
    "LLM_MODEL_VERSION", "meta/llama-4-scout-17b-16e-instruct-maas"
)
_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
_LOCATION = os.environ.get("GCP_LOCATION", "us-east5")
_API_KEY = os.environ.get("GEMINI_API_KEY")

_MAX_RETRIES = 5
_TIMEOUT_S = 60
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class LLMClientError(RuntimeError):
    """The LLM call failed, or its output didn't match the requested schema
    after all retries.
    """


def _endpoint_url() -> str:
    if not _PROJECT_ID:
        raise LLMClientError(
            "GCP_PROJECT_ID is not set. Required to call Gemini Enterprise "
            "Agent Platform — see .env.example."
        )
    return (
        f"https://{_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{_PROJECT_ID}/locations/{_LOCATION}/endpoints/openapi/chat/completions"
    )


def _access_token() -> str:
    """Fresh ADC access token.

    Not cached at module level on purpose: `Credentials.refresh` is cheap
    when the cached token still has headroom, and skipping our own caching
    avoids a class of bugs where a worker process holds an expired token
    across a long-running Celery task.
    """
    credentials, _ = google.auth.default(scopes=_SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def _auth_headers() -> dict[str, str]:
    """`GEMINI_API_KEY`, if set, skips ADC entirely — see module docstring
    for why this is a testing convenience, not the recommended path.
    """
    if _API_KEY:
        return {"x-goog-api-key": _API_KEY}
    return {"Authorization": f"Bearer {_access_token()}"}


_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _should_retry(exc: Exception) -> bool:
    """A 4xx like 404/401/403 (wrong URL, bad auth, license not accepted)
    won't fix itself by waiting — retrying just adds delay before the same
    failure repeats. Only rate limits, transient server errors, and
    connection-level problems (no response at all) are worth retrying.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        return response.status_code in _RETRYABLE_STATUS_CODES
    return True


def _retry_delay(exc: Exception, attempt: int) -> float:
    """429s need longer, jittered backoff than other errors.

    The debate graph fans Financial/Sentiment/Macro out concurrently (see
    graph.py), so all three can hit the MaaS quota at once — a short fixed
    backoff has them all retry in lockstep and hit it again together.
    Respects `Retry-After` if the API sends one; otherwise backs off harder
    than the default path and adds jitter to break that lockstep.
    """
    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(60.0, 5 * 2**attempt) + random.uniform(0, 1)
    return 2**attempt


def _call(body: dict, parse: Callable[[dict], R], label: str, detail: str = "") -> R:
    """POST `body`, hand the response JSON to `parse`, and return its result.

    One retry loop, one budget of `_MAX_RETRIES` attempts, covers both network
    errors and responses `parse` cannot read (the model may return a broken
    JSON). Only errors worth retrying are retried (see `_should_retry`) — a
    permanent error like a 404 fails immediately instead of burning through
    the full backoff schedule first.
    """
    url = _endpoint_url()
    what = f"{label}{detail}"
    logger.debug("prompt for %s:\n%s", what, body["messages"][0]["content"])
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=body,
                headers={**_auth_headers(), "Content-Type": "application/json"},
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
            result = parse(data)
            logger.debug("response for %s:\n%s", what, data["choices"][0]["message"])
            usage.record_call(
                what, data.get("usage"), time.monotonic() - started, attempt
            )
            return result
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            last_error = exc
            if not _should_retry(exc):
                break
            if attempt < _MAX_RETRIES:
                delay = _retry_delay(exc, attempt)
                usage.record_retry(what, attempt, _MAX_RETRIES, delay, exc)
                time.sleep(delay)

    usage.record_failure(what, time.monotonic() - started, attempt, last_error)
    raise LLMClientError(
        f"{label} failed after {attempt} attempt(s){detail}: {last_error}"
    ) from last_error


def _unwrap_function_call(answer: object) -> object:
    """Llama 4 sometimes wraps a structured answer as a function call:
    `{"type": "function", "name": ..., "parameters": {...}}`. Return the inner
    object in that case, the answer unchanged otherwise.
    """
    if not isinstance(answer, dict) or answer.get("type") != "function":
        return answer
    inner = answer.get("parameters", answer.get("arguments"))
    if isinstance(inner, str):
        inner = json.loads(inner)
    return inner if isinstance(inner, dict) else answer


def generate_structured(prompt: str, schema: type[T], temperature: float = 0.0) -> T:
    """Call the LLM with `prompt`, parse the response into `schema`.

    `temperature` defaults to 0 for deterministic, reproducible output —
    see docs/evaluation.md, backtests need results that don't change between
    runs of the same `as_of_date`/prompt.

    No `tools` param is sent here, and no grounding/search tool is ever
    attached to any call — see the module docstring and docs/evaluation.md §2
    for why that matters during backtests.
    """
    body = {
        "model": MODEL_VERSION,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        },
    }

    def parse(data: dict) -> T:
        content = data["choices"][0]["message"]["content"]
        return schema.model_validate(_unwrap_function_call(json.loads(content)))

    return _call(
        body, parse, "generate_structured", detail=f" for schema '{schema.__name__}'"
    )


@dataclass(frozen=True)
class ToolCall:
    """One function call the model asked for: its name and parsed arguments."""

    name: str
    arguments: dict


def _parse_tool_calls(data: dict) -> list[ToolCall]:
    message = data["choices"][0]["message"]
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:  # key missing or null -> []
        try:
            calls.append(
                ToolCall(
                    name=raw["function"]["name"],
                    # `arguments` is a JSON *string*, not a dict
                    arguments=json.loads(raw["function"]["arguments"]),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            logger.warning("Skipping malformed tool call: %r", raw)
    return calls


def generate_tool_calls(
    prompt: str, tools: list[dict], temperature: float = 0.0
) -> list[ToolCall]:
    """One call with `tools`. Returns the calls the model asked for, [] when it
    answered in plain text.

    The model never executes anything: it only asks. No `response_format` and
    no `tool_choice` on purpose — calling a tool is optional, and a plain
    answer is a valid outcome. `tools` must hold only our own local function
    tools (never a grounding/search tool, docs/evaluation.md §2).
    """
    body = {
        "model": MODEL_VERSION,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "tools": tools,
    }
    return _call(body, _parse_tool_calls, "generate_tool_calls")
