"""Pydantic mirrors of the backend contract (docs/ARCHITECTURE.md §4-5 and
src/cache/schemas.py), so the frontend never touches raw dicts.

Kept as a separate copy on purpose: the `frontend` container does not ship
`src/`, and the REST contract — not the Python classes — is the interface.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["BUY", "HOLD", "SELL"]
StatusValue = Literal["PENDING", "RUNNING", "DONE", "FAILED"]
AgentName = Literal["financial", "sentiment", "macro"]


class ApiError(RuntimeError):
    """Backend unreachable, returned an error, or sent an unexpected body."""


class AgentReport(BaseModel):
    opinion: str
    reasoning: str


class AgentReports(BaseModel):
    financial: AgentReport
    sentiment: AgentReport
    macro: AgentReport


class PMOpinion(BaseModel):
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    arguments: list[str] = []


class CriticFeedback(BaseModel):
    agree: bool
    feedback: str


class PredictResult(BaseModel):
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    rounds_used: int
    critic_agreed: bool
    agent_reports: AgentReports


class DebateStatus(BaseModel):
    status: StatusValue
    round_number: int = 1
    current_node: str | None = None  # last finished node
    updated_at: datetime | None = None
    error: str | None = None


class HistoryEntry(BaseModel):
    node: str
    round_number: int
    ts: datetime | None = None
    delta: dict[str, Any] = {}


class PredictResponse(BaseModel):
    """`GET /predict/result/{task_id}` — the full envelope from §5."""

    task_id: str
    ticker: str = ""
    horizon: str = ""
    status: StatusValue
    as_of_date: date | None = None
    model_version: str | None = None
    result: PredictResult | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
