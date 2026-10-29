"""Point-in-time read queries against the warehouse tables.

The read side of `src/db/loaders.py`: `loaders` writes the datalake into
Postgres, this reads it back out. Every query is bounded by `as_of_date` —
see `docs/project_decisions.md` for the inclusive-cutoff convention and
`docs/evaluation.md` §1 for why that bound is load-bearing.

Callers pass an explicit `Session`; nothing here opens or commits one.
"""

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Literal

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, defer

from src.db.models import ArticleSummary, Fundamentals, MacroSeries, StockPrice


# SEC filing deadline for a 10-K (10-Q: 40-45 days) — a safe upper bound.
REPORT_LAG_FALLBACK = timedelta(days=90)


def fetch_fundamentals(
    session: Session, ticker: str, as_of_date: date, quarters: int = 40
) -> list[Fundamentals]:
    """Rows for the last `quarters` distinct fiscal dates published by as_of_date.

    A report counts once `reported_date <= as_of_date`; without a reported
    date it is assumed published `REPORT_LAG_FALLBACK` after fiscal_date.

    Selects by distinct fiscal_date first, then fetches every report_type
    for those dates — a plain row-limit would cut a quarter's 3 report
    types (BALANCE_SHEET/CASH_FLOW/INCOME_STATEMENT) unevenly, returning an
    incomplete last quarter.
    """
    date_stmt = (
        select(Fundamentals.fiscal_date)
        .where(
            Fundamentals.symbol == ticker,
            func.coalesce(
                Fundamentals.reported_date,
                Fundamentals.fiscal_date + REPORT_LAG_FALLBACK,
            )
            <= as_of_date,
        )
        .distinct()
        .order_by(Fundamentals.fiscal_date.desc())
        .limit(quarters)
    )
    dates = session.scalars(date_stmt).all()
    if not dates:
        return []

    stmt = (
        select(Fundamentals)
        .where(Fundamentals.symbol == ticker, Fundamentals.fiscal_date.in_(dates))
        .order_by(Fundamentals.fiscal_date.desc(), Fundamentals.report_type)
    )
    return list(session.scalars(stmt))


def fetch_prices(
    session: Session, ticker: str, as_of_date: date, lookback_days: int = 3600
) -> list[StockPrice]:
    start = as_of_date - timedelta(days=lookback_days)
    stmt = (
        select(StockPrice)
        .where(
            StockPrice.symbol == ticker,
            StockPrice.date <= as_of_date,
            StockPrice.date >= start,
        )
        .order_by(StockPrice.date.desc())
    )
    return list(session.scalars(stmt))


def fetch_macro_series(
    session: Session, as_of_date: date, lookback_days: int = 3600
) -> list[MacroSeries]:
    """Rows for every series within the lookback window, <= as_of_date.

    The window is deliberately long (3600 days, ~10 years) so
    `format_macro_context` can show a long-run trend (latest vs. earliest
    value in the window) per series, not just a single snapshot. The
    trend computation itself happens there, not in SQL, so it stays
    unit-testable without a database.
    """
    start = as_of_date - timedelta(days=lookback_days)
    stmt = (
        select(MacroSeries)
        .where(MacroSeries.date <= as_of_date, MacroSeries.date >= start)
        .order_by(MacroSeries.series_id, MacroSeries.date.desc())
    )
    return list(session.scalars(stmt))


# The 768-float vectors are never needed by callers; no need to load them into memory.
_WITHOUT_EMBEDDINGS = (
    defer(ArticleSummary.summary_embedding),
    defer(ArticleSummary.title_summary_embedding),
)


def fetch_recent_articles(
    session: Session, ticker: str, as_of_date: date, limit: int = 10
) -> list[ArticleSummary]:
    """The `limit` most recent articles published on or before `as_of_date`.

    `time_published` is a timestamp, so `<= as_of_date` would coerce the bound
    to midnight and drop everything published on `as_of_date` itself.
    """
    stmt = (
        select(ArticleSummary)
        .options(*_WITHOUT_EMBEDDINGS)
        .where(
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
        )
        .order_by(ArticleSummary.time_published.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


EmbeddingVariant = Literal["summary", "title_summary"]

# Calibration: docs/project_decisions.md.
DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_VARIANT: EmbeddingVariant = "title_summary"


def fetch_similar_articles(
    session: Session,
    ticker: str,
    as_of_date: date,
    query_embedding: Sequence[float],
    limit: int = 10,
    variant: EmbeddingVariant = DEFAULT_VARIANT,
    min_similarity: float | None = DEFAULT_MIN_SIMILARITY,
) -> list[tuple[ArticleSummary, float]]:
    """Up to `limit` (article, cosine similarity) pairs, most similar first,
    published on or before `as_of_date`. `variant` picks the embedding column;
    articles without it, or below `min_similarity`, are skipped.
    """
    column = (
        ArticleSummary.summary_embedding
        if variant == "summary"
        else ArticleSummary.title_summary_embedding
    )
    distance = column.cosine_distance(list(query_embedding))
    stmt = (
        select(ArticleSummary, (1 - distance).label("similarity"))
        .options(*_WITHOUT_EMBEDDINGS)
        .where(
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
            column.is_not(None),
        )
        .order_by(distance)
        .limit(limit)
    )
    if min_similarity is not None:
        stmt = stmt.where(distance <= 1 - min_similarity)
    return [(article, float(sim)) for article, sim in session.execute(stmt)]


def fetch_articles_by_ids(
    session: Session, ticker: str, as_of_date: date, ids: Sequence[int]
) -> list[ArticleSummary]:
    """Articles found in earlier rounds, newest first.

    The debate state keeps only ids (small history deltas), so the text is read
    back here. Ticker and date are filtered again, defensively, like every
    other query in this module.
    """
    if not ids:
        return []
    stmt = (
        select(ArticleSummary)
        .options(*_WITHOUT_EMBEDDINGS)
        .where(
            ArticleSummary.id.in_(ids),
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
        )
        .order_by(ArticleSummary.time_published.desc())
    )
    return list(session.scalars(stmt))


def count_articles_without_embedding(
    session: Session,
    ticker: str,
    as_of_date: date,
    variant: EmbeddingVariant = DEFAULT_VARIANT,
) -> int:
    """Articles that have text but no vector — `fetch_similar_articles` skips
    them silently, so a non-zero count means retrieval sees a partial corpus
    (an unfinished `python -m src.db.load`).
    """
    if variant == "summary":
        column = ArticleSummary.summary_embedding
        has_text = ArticleSummary.summary != ""
    else:
        column = ArticleSummary.title_summary_embedding
        has_text = or_(ArticleSummary.summary != "", ArticleSummary.title != "")
    stmt = (
        select(func.count())
        .select_from(ArticleSummary)
        .where(
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
            column.is_(None),
            has_text,
        )
    )
    return session.scalar(stmt) or 0


def fetch_price_date_range(session: Session) -> tuple[date | None, date | None]:
    """First and last date in stock_prices (None, None when empty)."""
    first, last = session.execute(
        select(func.min(StockPrice.date), func.max(StockPrice.date))
    ).one()
    return first, last


def fetch_tickers(session: Session) -> list[str]:
    """Tickers that have price data, sorted."""
    stmt = select(StockPrice.symbol).distinct().order_by(StockPrice.symbol)
    return list(session.scalars(stmt))
