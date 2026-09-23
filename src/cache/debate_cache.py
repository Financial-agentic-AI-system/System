"""Redis-backed cache for live debate status and history.

Written by src/worker/tasks.py while iterating compiled_graph.stream();
read by the debate_status/history/predict_result endpoints in
src/api/v1/. Cache only, per docs/ARCHITECTURE.md §2 — TTL-bound, not a
permanent store.
"""

from datetime import datetime, timezone

from src.agents.state import PredictResult
from src.cache.redis_client import redis_client
from src.cache.schemas import DebateStatus, DebateStatusValue, HistoryEntry

CACHE_TTL_SECONDS = 60 * 60 * 24  # 24h — cache, not permanent storage


def _status_key(task_id: str) -> str:
    return f"debate:{task_id}:status"


def _history_key(task_id: str) -> str:
    return f"debate:{task_id}:history"


def _result_key(task_id: str) -> str:
    return f"debate:{task_id}:result"


def set_status(
    task_id: str,
    status: DebateStatusValue,
    round_number: int,
    current_node: str | None = None,
    error: str | None = None,
) -> None:
    payload = DebateStatus(
        status=status,
        round_number=round_number,
        current_node=current_node,
        updated_at=datetime.now(timezone.utc),
        error=error,
    )
    redis_client.set(
        _status_key(task_id), payload.model_dump_json(), ex=CACHE_TTL_SECONDS
    )


def get_status(task_id: str) -> DebateStatus | None:
    raw = redis_client.get(_status_key(task_id))
    if raw is None:
        return None
    return DebateStatus.model_validate_json(raw)


def append_history(task_id: str, node: str, round_number: int, delta: dict) -> None:
    entry = HistoryEntry(
        node=node,
        round_number=round_number,
        ts=datetime.now(timezone.utc),
        delta=delta,
    )
    key = _history_key(task_id)
    redis_client.rpush(key, entry.model_dump_json())
    redis_client.expire(key, CACHE_TTL_SECONDS)


def get_history(task_id: str) -> list[HistoryEntry]:
    raw_list = redis_client.lrange(_history_key(task_id), 0, -1)
    return [HistoryEntry.model_validate_json(raw) for raw in raw_list]


def set_result(task_id: str, result: PredictResult) -> None:
    redis_client.set(
        _result_key(task_id), result.model_dump_json(), ex=CACHE_TTL_SECONDS
    )


def get_result(task_id: str) -> PredictResult | None:
    raw = redis_client.get(_result_key(task_id))
    if raw is None:
        return None
    return PredictResult.model_validate_json(raw)
