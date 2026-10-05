"""Unit tests for src/retriever/embeddings.py — HTTP and auth are stubbed."""

import google.auth.exceptions
import pytest
import requests

from src.retriever import embeddings

DIM = embeddings.EMBEDDING_DIM


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._payload


def _ok(n: int, dim: int = DIM) -> FakeResponse:
    return FakeResponse(
        payload={
            "predictions": [
                {"embeddings": {"values": [float(i)] * dim}} for i in range(n)
            ]
        }
    )


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.setattr(embeddings, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    calls: list[tuple[str, dict]] = []
    queue: list[FakeResponse] = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json))
        return queue.pop(0) if queue else _ok(len(json["instances"]))

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    return queue, calls


def test_request_shape_and_endpoint(http):
    _, calls = http

    vectors = embeddings.embed_texts(["a", "b"])

    assert len(vectors) == 2 and len(vectors[0]) == DIM
    url, body = calls[0]
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/"
        "us-central1/publishers/google/models/text-embedding-005:predict"
    )
    assert body["instances"] == [
        {"content": "a", "task_type": "RETRIEVAL_DOCUMENT"},
        {"content": "b", "task_type": "RETRIEVAL_DOCUMENT"},
    ]
    assert body["parameters"]["outputDimensionality"] == DIM


def test_texts_are_split_into_batches_in_order(http, monkeypatch):
    _, calls = http
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")

    vectors = embeddings.embed_texts(["a", "b", "c", "d", "e"])

    assert [len(body["instances"]) for _, body in calls] == [2, 2, 1]
    assert len(vectors) == 5


def test_gemini_embedding_sends_one_text_per_request(http, monkeypatch):
    _, calls = http
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")

    embeddings.embed_texts(["a", "b", "c"])

    assert len(calls) == 3
    assert "gemini-embedding-001:predict" in calls[0][0]


def test_query_uses_retrieval_query_task_type(http):
    _, calls = http
    embeddings.embed_query("is the news positive?")
    assert calls[0][1]["instances"][0]["task_type"] == "RETRIEVAL_QUERY"


def test_transient_error_is_retried(http):
    queue, calls = http
    queue += [FakeResponse(503), _ok(1)]
    assert len(embeddings.embed_texts(["a"])) == 1
    assert len(calls) == 2


def test_permanent_error_fails_without_retrying(http):
    queue, calls = http
    queue.append(FakeResponse(403, text="permission denied"))
    with pytest.raises(embeddings.EmbeddingError, match="permission denied"):
        embeddings.embed_texts(["a"])
    assert len(calls) == 1


def test_wrong_dimension_is_rejected(http):
    queue, _ = http
    queue.append(_ok(1, dim=3))
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_texts(["a"])


def test_missing_project_id(monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    assert embeddings.is_configured() is False
    with pytest.raises(embeddings.EmbeddingError, match="GCP_PROJECT_ID"):
        embeddings.embed_texts(["a"])


class _FakeCredentials:
    def __init__(self, fail_times: int = 0, error: Exception | None = None) -> None:
        self.valid = False
        self.token = None
        self.refreshes = 0
        self._fail_times = fail_times
        self._error = error

    def refresh(self, request) -> None:
        self.refreshes += 1
        if self.refreshes <= self._fail_times:
            raise self._error
        self.valid = True
        self.token = "tok"


@pytest.fixture
def adc(monkeypatch, http):
    """Real `_auth_headers` (undo the stub from `http`) over fake credentials."""
    monkeypatch.undo()
    monkeypatch.setenv("GCP_PROJECT_ID", "proj")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1")
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    monkeypatch.setattr(embeddings, "_credentials", None)
    calls: list[dict] = []

    def fake_post(url, json, headers, timeout):
        calls.append(headers)
        return _ok(len(json["instances"]))

    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    def install(credentials: _FakeCredentials) -> list[dict]:
        monkeypatch.setattr(
            embeddings.google.auth, "default", lambda scopes: (credentials, "proj")
        )
        return calls

    return install


def test_token_is_fetched_once_for_many_batches(adc):
    credentials = _FakeCredentials()
    calls = adc(credentials)

    embeddings.embed_texts(["a", "b", "c"])

    assert credentials.refreshes == 1
    assert [h["Authorization"] for h in calls] == ["Bearer tok"] * 3


def test_dropped_connection_during_token_refresh_is_retried(adc):
    error = google.auth.exceptions.TransportError("Connection aborted.")
    credentials = _FakeCredentials(fail_times=2, error=error)
    adc(credentials)

    assert len(embeddings.embed_texts(["a"])) == 1
    assert credentials.refreshes == 3


def test_rejected_credentials_fail_fast_with_a_hint(adc):
    error = google.auth.exceptions.RefreshError("invalid_grant")
    credentials = _FakeCredentials(fail_times=99, error=error)
    adc(credentials)

    with pytest.raises(embeddings.EmbeddingError, match="application-default login"):
        embeddings.embed_texts(["a"])
    assert credentials.refreshes == 1
