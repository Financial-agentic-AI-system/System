"""`GET /tickers` — the tickers the frontend can offer."""

from fastapi import APIRouter

from src.api.v1.schemas import TickersResponse
from src.db.queries import fetch_tickers
from src.db.session import get_session

router = APIRouter(tags=["tickers"])


@router.get("/tickers", response_model=TickersResponse)
def get_tickers() -> TickersResponse:
    with get_session() as session:
        return TickersResponse(tickers=fetch_tickers(session))
