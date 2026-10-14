"""`GET /meta` — settings the frontend builds its forms from."""

from fastapi import APIRouter

from src.agents.state import MAX_ROUNDS
from src.api.v1.schemas import (
    BACKTEST_END,
    BACKTEST_START,
    HORIZONS,
    MetaResponse,
)

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    return MetaResponse(
        horizons=HORIZONS,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        max_rounds=MAX_ROUNDS,
    )
