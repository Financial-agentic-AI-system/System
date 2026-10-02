"""REST contract of the backend (docs/ARCHITECTURE.md §4-5).

One Request/Response pair per endpoint group, per CLAUDE.md. The debate
status and history entries reuse the Redis wire shapes from
`src/cache/schemas.py` — the endpoints are plain reads off that cache, so
there is nothing to translate.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.agents.state import PredictResult
from src.cache.schemas import DebateStatus, DebateStatusValue, HistoryEntry

Horizon = Literal["1W", "1M", "3M"]


class PredictStartRequest(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z0-9.\-]{1,16}$", examples=["TSLA"])
    horizon: Horizon = Field(examples=["1W"])
    as_of_date: date = Field(
        description=(
            "Point-in-time anchor: agents only see data up to this date "
            "(docs/evaluation.md §3.1)."
        ),
        examples=["2026-01-15"],
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("as_of_date")
    @classmethod
    def _not_in_the_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("as_of_date cannot be in the future")
        return value


class PredictStartResponse(BaseModel):
    task_id: str
    status: DebateStatusValue = "PENDING"


class PredictResultResponse(BaseModel):
    """`GET /predict/result/{task_id}` — the §5 envelope. `result` is only set
    once `status` is DONE; `error` only when it is FAILED.
    """

    task_id: str
    ticker: str
    horizon: str
    status: DebateStatusValue
    as_of_date: date
    model_version: str
    result: PredictResult | None = None
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class DebateStatusResponse(DebateStatus):
    """Same fields as the cached `DebateStatus`."""


class HistoryResponse(BaseModel):
    task_id: str
    history: list[HistoryEntry]
