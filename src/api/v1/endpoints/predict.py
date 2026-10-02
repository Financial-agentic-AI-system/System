"""`POST /predict/start` and `GET /predict/result/{task_id}`."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from src.agents.llm_client import MODEL_VERSION
from src.api.v1.schemas import (
    PredictResultResponse,
    PredictStartRequest,
    PredictStartResponse,
)
from src.cache import debate_cache
from src.cache.schemas import TaskMeta
from src.worker.celery_app import app as celery_app

# Dispatch by name, not by importing `src.worker.tasks`: importing it would
# pull the whole LangGraph/DB stack into the API process just to enqueue.
RUN_DEBATE_TASK = "src.worker.tasks.run_debate"

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/start", response_model=PredictStartResponse, status_code=202)
def start_prediction(request: PredictStartRequest) -> PredictStartResponse:
    """Enqueue a debate and return its `task_id` immediately."""
    task_id = uuid4().hex

    # Written *before* dispatch so a client polling straight away never gets a
    # 404 for a task that was accepted.
    debate_cache.set_meta(
        task_id,
        TaskMeta(
            ticker=request.ticker,
            horizon=request.horizon,
            as_of_date=request.as_of_date,
            model_version=MODEL_VERSION,
            created_at=datetime.now(timezone.utc),
        ),
    )
    debate_cache.set_status(task_id, "PENDING", round_number=1)

    try:
        celery_app.send_task(
            RUN_DEBATE_TASK,
            # ISO string on purpose: the worker converts it back (tasks.py).
            args=[request.ticker, request.horizon, request.as_of_date.isoformat()],
            task_id=task_id,
        )
    except Exception as exc:
        # Broker unreachable: don't leave a task that stays PENDING forever.
        debate_cache.set_status(
            task_id, "FAILED", round_number=1, error=f"Could not enqueue task: {exc}"
        )
        raise

    return PredictStartResponse(task_id=task_id, status="PENDING")


@router.get("/result/{task_id}", response_model=PredictResultResponse)
def get_prediction_result(task_id: str) -> PredictResultResponse:
    """Poll the prediction. `result` is null until the debate is DONE."""
    meta = debate_cache.get_meta(task_id)
    status = debate_cache.get_status(task_id)
    if meta is None or status is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown task_id (debates are kept in Redis for 24 h only).",
        )

    result = None
    if status.status == "DONE":
        result = debate_cache.get_result(task_id)
        if result is None:
            raise HTTPException(
                status_code=404, detail="The debate is done but its result expired."
            )

    finished = status.status in {"DONE", "FAILED"}
    return PredictResultResponse(
        task_id=task_id,
        ticker=meta.ticker,
        horizon=meta.horizon,
        status=status.status,
        as_of_date=meta.as_of_date,
        model_version=meta.model_version,
        result=result,
        created_at=meta.created_at,
        # the last status write is the DONE/FAILED one
        finished_at=status.updated_at if finished else None,
        error=status.error,
    )
