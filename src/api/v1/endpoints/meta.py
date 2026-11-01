"""`GET /meta` — settings the frontend builds its forms from."""

from fastapi import APIRouter

from src.agents.state import MAX_ROUNDS
from src.api.v1.schemas import (
    BACKTEST_END,
    BACKTEST_START,
    HORIZONS,
    MetaResponse,
)
from src.db.queries import fetch_price_date_range
from src.db.session import get_session

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    with get_session() as session:
        first, last = fetch_price_date_range(session)
    start = max(first, BACKTEST_START) if first else BACKTEST_START
    end = last if last else BACKTEST_END
    return MetaResponse(
        horizons=HORIZONS,
        backtest_start=start,
        backtest_end=max(end, start),
        max_rounds=MAX_ROUNDS,
    )
