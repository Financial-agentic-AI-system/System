"""Shape of what's stored in Redis by src/cache/debate_cache.py.

Distinct from src/agents/state.py (in-graph state) and the future
src/api/v1/schemas.py (REST contract) — this is the on-the-wire shape for
the `debate:{task_id}:*` keys specifically.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

DebateStatusValue = Literal["PENDING", "RUNNING", "DONE", "FAILED"]


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
