"""Text embeddings from Gemini Enterprise Agent Platform (Vertex AI).

Default model: `text-embedding-005` (768 dims, batched requests).
`gemini-embedding-001` also works via EMBEDDING_MODEL, one text per request.

Auth as in `src/agents/llm_client.py` (ADC or GEMINI_API_KEY). Env vars:
GCP_PROJECT_ID (required), GCP_LOCATION, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE.
"""

import os
import time
from collections.abc import Sequence
from typing import Literal

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
import requests

from src.agents.llm_client import _retry_delay, _should_retry

DEFAULT_MODEL = "text-embedding-005"
# Size of the `vector` columns in src/db/models.py.
EMBEDDING_DIM = 768

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_DEFAULT_BATCH_SIZE = 50  # API limit is 250 texts / 20k tokens per request
_MAX_RETRIES = 5
_TIMEOUT_S = 60
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class EmbeddingError(RuntimeError):
    """The embedding call failed or returned an unexpected body."""


def model_name() -> str:
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)


def is_configured() -> bool:
    """False when GCP_PROJECT_ID is missing, so callers can skip embedding."""
    return bool(os.environ.get("GCP_PROJECT_ID"))


def batch_size() -> int:
    """Texts per request. `gemini-embedding-*` accepts exactly one."""
    if model_name().startswith("gemini-embedding"):
        return 1
    return int(os.environ.get("EMBEDDING_BATCH_SIZE", _DEFAULT_BATCH_SIZE))


def _endpoint_url() -> str:
    project = os.environ.get("GCP_PROJECT_ID")
    if not project:
        raise EmbeddingError(
            "GCP_PROJECT_ID is not set. Required to call the embedding model — "
            "see .env.example."
        )
    location = os.environ.get("GCP_LOCATION", "us-central1")
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/publishers/google/models/{model_name()}:predict"
    )


_credentials = None


def _auth_headers() -> dict[str, str]:
    """ADC token, cached and refreshed only when missing or expired."""
    global _credentials
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return {"x-goog-api-key": api_key}
    if _credentials is None:
        _credentials, _ = google.auth.default(scopes=_SCOPES)
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
    return {"Authorization": f"Bearer {_credentials.token}"}


def _embed_batch(texts: Sequence[str], task_type: TaskType) -> list[list[float]]:
    body = {
        "instances": [{"content": text, "task_type": task_type} for text in texts],
        "parameters": {"autoTruncate": True, "outputDimensionality": EMBEDDING_DIM},
    }
    url = _endpoint_url()
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
            vectors = [
                prediction["embeddings"]["values"]
                for prediction in response.json()["predictions"]
            ]
        except google.auth.exceptions.RefreshError as exc:
            raise EmbeddingError(
                "Google credentials were rejected. Run "
                f"`gcloud auth application-default login` and retry. ({exc})"
            ) from exc
        except (
            requests.RequestException,
            google.auth.exceptions.TransportError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            if not _should_retry(exc):
                break
            if attempt < _MAX_RETRIES:
                time.sleep(_retry_delay(exc, attempt))
            continue
        if len(vectors) != len(texts) or any(
            len(vector) != EMBEDDING_DIM for vector in vectors
        ):
            raise EmbeddingError(
                f"{model_name()} returned {len(vectors)} vector(s) for "
                f"{len(texts)} text(s), or not {EMBEDDING_DIM}-dimensional ones."
            )
        return vectors

    detail = getattr(getattr(last_error, "response", None), "text", "")
    raise EmbeddingError(
        f"Embedding call to {model_name()} failed after {attempt} attempt(s): "
        f"{last_error} {detail}".strip()
    ) from last_error


def embed_texts(
    texts: Sequence[str], task_type: TaskType = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """One 768-dim vector per text, in order. Use `RETRIEVAL_DOCUMENT` for
    stored articles and `RETRIEVAL_QUERY` (`embed_query`) for search text.
    """
    size = batch_size()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        vectors += _embed_batch(texts[start : start + size], task_type)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
