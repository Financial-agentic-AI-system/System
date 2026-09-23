"""Shape of what's stored in Redis by src/cache/debate_cache.py.

Distinct from src/agents/state.py (in-graph state) and the future
src/api/v1/schemas.py (REST contract) — this is the on-the-wire shape for
the `debate:{task_id}:*` keys specifically.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel

DebateStatusValue = Literal["PENDING", "RUNNING", "DONE", "FAILED"]


class TaskMeta(BaseModel):
    """What the client asked for — written once by POST /predict/start.

    The worker's own keys (status/history/result) don't carry the request, but
    the `/predict/result` envelope (docs/ARCHITECTURE.md §5) needs ticker,
    horizon, as_of_date and the model version.
    """

    ticker: str
    horizon: str
    as_of_date: date
    model_version: str
    created_at: datetime


class DebateStatus(BaseModel):
    status: DebateStatusValue
    round_number: int
    current_node: str | None = None  # last finished node
    updated_at: datetime
    error: str | None = None


class HistoryEntry(BaseModel):
    node: str
    round_number: int
    ts: datetime
    delta: dict[str, Any]
