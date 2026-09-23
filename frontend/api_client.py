"""REST client for the FastAPI backend (docs/ARCHITECTURE.md §4).

`BackendClient` and `mock_backend.MockClient` expose the same four methods,
so pages never care which one they talk to — see `get_client()`.
"""

from datetime import date
from typing import Any

import config
import requests
from models import (
    ApiError,
    DebateStatus,
    HistoryEntry,
    PredictResponse,
    PredictResult,
)
from pydantic import ValidationError


class BackendClient:
    def __init__(self, base_url: str = config.API_URL) -> None:
        self._base = f"{base_url}{config.API_PREFIX}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}{path}"
        try:
            resp = requests.request(
                method, url, timeout=config.REQUEST_TIMEOUT_S, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach the backend at {url}: {exc}") from exc
        if resp.status_code == 404:
            raise ApiError("Unknown task_id (results expire from Redis after 24 h).")
        if not resp.ok:
            raise ApiError(
                f"Backend returned {resp.status_code} for {url}: {resp.text}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ApiError(f"Backend returned a non-JSON body for {url}") from exc

    @staticmethod
    def _parse(model: type, payload: Any) -> Any:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ApiError(f"Unexpected response shape: {exc}") from exc

    def start(self, ticker: str, horizon: str, as_of_date: date) -> str:
        body = {
            "ticker": ticker,
            "horizon": horizon,
            "as_of_date": as_of_date.isoformat(),
        }
        payload = self._request("POST", "/predict/start", json=body)
        try:
            return str(payload["task_id"])
        except (KeyError, TypeError) as exc:
            raise ApiError("Response of /predict/start has no task_id") from exc

    def status(self, task_id: str) -> DebateStatus:
        payload = self._request("GET", f"/debate_status/{task_id}")
        return self._parse(DebateStatus, payload)

    def history(self, task_id: str) -> list[HistoryEntry]:
        payload = self._request("GET", f"/history/{task_id}")
        if isinstance(payload, dict):  # tolerate {"history": [...]}
            payload = payload.get("history", [])
        return [self._parse(HistoryEntry, item) for item in payload]

    def result(self, task_id: str) -> PredictResponse:
        payload = self._request("GET", f"/predict/result/{task_id}")
        if isinstance(payload, dict) and "direction" in payload:
            # Redis stores the bare PredictResult (src/cache/debate_cache.py);
            # wrap it so callers always get the §5 envelope.
            payload = {
                "task_id": task_id,
                "status": "DONE",
                "result": self._parse(PredictResult, payload).model_dump(),
            }
        return self._parse(PredictResponse, payload)


def get_client(mock: bool):
    """Return the mock engine or the real REST client."""
    if mock:
        from mock_backend import get_mock_client

        return get_mock_client()
    return BackendClient()
