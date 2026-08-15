"""LLM client for Gemini Enterprise Agent Platform — Llama 3.3 70B Instruct
via MaaS. Cutoff (Dec 2023) and no-grounding rationale: docs/evaluation.md
§§2-3.3. Update that doc's "Model + version" row if this changes.

Auth: ADC by default (`gcloud auth application-default login`). Set
GEMINI_API_KEY to use an API key instead — untested against this specific
endpoint, treat as a local-testing convenience, not the recommended path.

Env vars (see .env.example): GCP_PROJECT_ID (required), GCP_LOCATION
(default us-central1), LLM_MODEL_VERSION, GEMINI_API_KEY (optional).

Requires accepting the Llama Community License Agreement on the model's
Model Garden page once per project, or every call 404s.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import TypeVar

import google.auth
import google.auth.transport.requests
import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

MODEL_VERSION = os.environ.get("LLM_MODEL_VERSION", "meta/llama-3.3-70b-instruct-maas")
_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
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


def generate_structured(prompt: str, schema: type[T]) -> T:
    """Call the LLM with `prompt`, parse the response into `schema`.

    No `tools` param is ever sent to the API — see the module docstring and
    docs/evaluation.md §2 for why that matters during backtests.

    Retries up to `_MAX_RETRIES` times, but only for errors worth retrying
    (see `_should_retry`) — a permanent error like a 404 fails immediately
    instead of burning through the full backoff schedule first.
    """
    url = _endpoint_url()
    body = {
        "model": MODEL_VERSION,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        },
    }

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
            content = response.json()["choices"][0]["message"]["content"]
            return schema.model_validate(json.loads(content))
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
                time.sleep(_retry_delay(exc, attempt))

    raise LLMClientError(
        f"generate_structured failed after {attempt} attempt(s) for schema "
        f"'{schema.__name__}': {last_error}"
    ) from last_error
