"""ORM models for the datalake data.

Per the design decision, financial statements (BALANCE_SHEET, CASH_FLOW,
INCOME_STATEMENT) go into a single hybrid table: key columns (symbol,
report_type, period, fiscal_date) plus the whole report in a JSONB column.
The other datasets have a simple, stable shape and get fully typed tables.
"""

import datetime as dt

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Fundamentals(Base):
    """A quarterly/annual financial report (payload stored in JSONB)."""

    __tablename__ = "fundamentals"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "report_type",
            "period",
            "fiscal_date",
            name="uq_fundamentals_key",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    report_type: Mapped[str] = mapped_column(String(32), index=True)
    period: Mapped[str] = mapped_column(String(16))
    fiscal_date: Mapped[dt.date | None] = mapped_column(Date, index=True)
    data: Mapped[dict] = mapped_column(JSONB)


class StockPrice(Base):
    """A daily closing price."""

    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_stock_price_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    close: Mapped[float] = mapped_column(Numeric(20, 6))


class ArticleSummary(Base):
    """A news article summary with sentiment scores."""

    __tablename__ = "article_summaries"
    __table_args__ = (
        UniqueConstraint("symbol", "url", name="uq_article_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str | None] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    time_published: Mapped[dt.datetime | None] = mapped_column(DateTime)
    source: Mapped[str | None] = mapped_column(String)
    summary: Mapped[str | None] = mapped_column(String)
    overall_sentiment_score: Mapped[float | None] = mapped_column(Float)
    overall_sentiment_label: Mapped[str | None] = mapped_column(String(32))
    ticker_sentiment_score: Mapped[float | None] = mapped_column(Float)
    ticker_sentiment_label: Mapped[str | None] = mapped_column(String(32))
    relevance_score: Mapped[float | None] = mapped_column(Float)


class MacroSeries(Base):
    """A single point of a FRED macro series (series_id + date -> value)."""

    __tablename__ = "macro_series"
    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uq_macro_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    series_id: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    value: Mapped[float] = mapped_column(Float)
