"""`GET /debate_status/{task_id}` and `GET /history/{task_id}` — plain reads
off the Redis cache that the worker fills while the graph streams
(docs/ARCHITECTURE.md §6).
"""

from fastapi import APIRouter, HTTPException

from src.api.v1.schemas import DebateStatusResponse, HistoryResponse
from src.cache import debate_cache

router = APIRouter(tags=["debate"])

_UNKNOWN_TASK = "Unknown task_id (debates are kept in Redis for 24 h only)."


@router.get("/debate_status/{task_id}", response_model=DebateStatusResponse)
def get_debate_status(task_id: str) -> DebateStatusResponse:
    status = debate_cache.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=_UNKNOWN_TASK)
    return DebateStatusResponse(**status.model_dump())


@router.get("/history/{task_id}", response_model=HistoryResponse)
def get_history(task_id: str) -> HistoryResponse:
    """Transcript so far, one entry per finished node, in order. A task that
    is queued but has produced nothing yet returns an empty list.
    """
    if debate_cache.get_status(task_id) is None:
        raise HTTPException(status_code=404, detail=_UNKNOWN_TASK)
    return HistoryResponse(task_id=task_id, history=debate_cache.get_history(task_id))
