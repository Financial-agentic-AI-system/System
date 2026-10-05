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

from sqlalchemy import select
from sqlalchemy.orm import Session, defer

from src.db.models import ArticleSummary, Fundamentals, MacroSeries, StockPrice


def fetch_fundamentals(
    session: Session, ticker: str, as_of_date: date, quarters: int = 40
) -> list[Fundamentals]:
    """Rows for the last `quarters` distinct fiscal dates <= as_of_date.

    Selects by distinct fiscal_date first, then fetches every report_type
    for those dates — a plain row-limit would cut a quarter's 3 report
    types (BALANCE_SHEET/CASH_FLOW/INCOME_STATEMENT) unevenly, returning an
    incomplete last quarter.
    """
    date_stmt = (
        select(Fundamentals.fiscal_date)
        .where(Fundamentals.symbol == ticker, Fundamentals.fiscal_date <= as_of_date)
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


def fetch_recent_articles(
    session: Session, ticker: str, as_of_date: date, limit: int = 10
) -> list[ArticleSummary]:
    """The `limit` most recent articles published on or before `as_of_date`.

    `time_published` is a timestamp, so `<= as_of_date` would coerce the bound
    to midnight and drop everything published on `as_of_date` itself.
    """
    stmt = (
        select(ArticleSummary)
        .where(
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
        )
        .order_by(ArticleSummary.time_published.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


EmbeddingVariant = Literal["summary", "title_summary"]


def fetch_similar_articles(
    session: Session,
    ticker: str,
    as_of_date: date,
    query_embedding: Sequence[float],
    limit: int = 10,
    variant: EmbeddingVariant = "summary",
) -> list[ArticleSummary]:
    """The `limit` articles closest (cosine) to `query_embedding`, published
    on or before `as_of_date`. `variant` picks which embedding column is
    searched; articles without that embedding are skipped.

    Same day-inclusive bound as `fetch_recent_articles`. The query vector
    comes from `src.retriever.embeddings.embed_query`.
    """
    column = (
        ArticleSummary.summary_embedding
        if variant == "summary"
        else ArticleSummary.title_summary_embedding
    )
    stmt = (
        select(ArticleSummary)
        .options(
            defer(ArticleSummary.summary_embedding),
            defer(ArticleSummary.title_summary_embedding),
        )
        .where(
            ArticleSummary.symbol == ticker,
            ArticleSummary.time_published < as_of_date + timedelta(days=1),
            column.is_not(None),
        )
        .order_by(column.cosine_distance(list(query_embedding)))
        .limit(limit)
    )
    return list(session.scalars(stmt))
