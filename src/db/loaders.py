"""Load JSON/CSV data from the datalake into Postgres.

Separation of concerns:
- `clean_*` / `parse_*` / `iter_*` functions are pure (no DB) and testable,
- `load_*` functions perform an idempotent upsert into the database.

The datalake is read-only — nothing here writes back to files.
"""

import csv
import datetime as dt
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.db.models import ArticleSummary, Fundamentals, MacroSeries, StockPrice

# raw_data subdirectories treated as financial reports (hybrid schema).
FUNDAMENTAL_REPORTS: tuple[str, ...] = (
    "BALANCE_SHEET",
    "CASH_FLOW",
    "INCOME_STATEMENT",
)

_NULL_STRINGS: frozenset[str] = frozenset({"None", "", "-", "."})
_INT_RE = re.compile(r"-?\d+$")


# --------------------------------------------------------------------------- #
# Datalake path resolution
# --------------------------------------------------------------------------- #
def resolve_raw_data(datalake: str | None = None) -> Path:
    """Return the path to the `raw_data` directory inside the datalake.

    Lookup order: argument -> env DATALAKE_PATH -> common locations
    (container `/datalake`, sibling `../datalake` / `../Datalake`).
    """
    candidates: list[Path] = []
    if datalake:
        candidates.append(Path(datalake))
    env = os.getenv("DATALAKE_PATH")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("/datalake"),
        Path("../datalake"),
        Path("../Datalake"),
    ]

    for base in candidates:
        raw = base / "raw_data"
        if raw.is_dir():
            return raw
        if base.name == "raw_data" and base.is_dir():
            return base

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find a raw_data directory in the datalake. Tried: {tried}"
    )


# --------------------------------------------------------------------------- #
# Value cleaning and parsing
# --------------------------------------------------------------------------- #
def clean_value(value: object) -> object:
    """Normalize a raw value coming from Alpha Vantage.

    - the strings "None"/""/"-"/"." -> None,
    - numeric strings -> int or float (so JSONB queries are easier),
    - dates/currencies/other strings are left unchanged.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped in _NULL_STRINGS:
        return None
    if _INT_RE.match(stripped):
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return stripped


def clean_report(report: dict) -> dict:
    """Return a copy of the report with normalized values."""
    return {key: clean_value(val) for key, val in report.items()}


def parse_date(value: str | None) -> dt.date | None:
    """Parse 'YYYY-MM-DD' -> date (None when empty/invalid)."""
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def parse_time_published(value: str | None) -> dt.datetime | None:
    """Parse the Alpha Vantage '20250101T093000' format -> datetime."""
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _to_float(value: object) -> float | None:
    """Convert to float, tolerating None / non-numeric strings."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Record extraction (pure iterators: file -> rows)
# --------------------------------------------------------------------------- #
def iter_fundamentals(raw_data: Path) -> Iterator[dict]:
    """Rows for the `fundamentals` table from BS/CF/INCOME_STATEMENT."""
    for report_type in FUNDAMENTAL_REPORTS:
        directory = raw_data / report_type
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            symbol = payload.get("symbol") or path.stem.split("_")[0]
            period = payload.get("period") or "unknown"
            for report in payload.get("reports", []):
                cleaned = clean_report(report)
                yield {
                    "symbol": symbol,
                    "report_type": report_type,
                    "period": period,
                    "fiscal_date": parse_date(report.get("fiscalDateEnding")),
                    "data": cleaned,
                }


def iter_stock_prices(raw_data: Path) -> Iterator[dict]:
    """Rows for the `stock_prices` table."""
    directory = raw_data / "stock_price"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        symbol = payload.get("symbol") or path.stem.split("_")[0]
        for report in payload.get("reports", []):
            date = parse_date(report.get("date"))
            close = _to_float(report.get("close"))
            if date is None or close is None:
                continue
            yield {"symbol": symbol, "date": date, "close": close}


def iter_article_summaries(raw_data: Path) -> Iterator[dict]:
    """Rows for the `article_summaries` table."""
    directory = raw_data / "ARTICLE_SUMMARIES"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        symbol = path.stem.split("_")[0]
        articles = payload if isinstance(payload, list) else payload.get("feed", [])
        for article in articles:
            url = article.get("url")
            if not url:
                continue
            yield {
                "symbol": symbol,
                "title": article.get("title"),
                "url": url,
                "time_published": parse_time_published(article.get("time_published")),
                "source": article.get("source"),
                "summary": article.get("summary"),
                "overall_sentiment_score": _to_float(
                    article.get("overall_sentiment_score")
                ),
                "overall_sentiment_label": article.get("overall_sentiment_label"),
                "ticker_sentiment_score": _to_float(
                    article.get("ticker_sentiment_score")
                ),
                "ticker_sentiment_label": article.get("ticker_sentiment_label"),
                "relevance_score": _to_float(article.get("relevance_score")),
            }


def iter_macro_series(raw_data: Path) -> Iterator[dict]:
    """Rows for the `macro_series` table from FRED CSV files."""
    directory = raw_data / "FRED_MACRO"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.csv")):
        series_id = path.stem
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header: DATE,<SERIES>
            for row in reader:
                if len(row) < 2:
                    continue
                date = parse_date(row[0])
                value = _to_float(row[1])
                if date is None or value is None:
                    continue
                yield {"series_id": series_id, "date": date, "value": value}


# --------------------------------------------------------------------------- #
# Upsert into the database (idempotent)
# --------------------------------------------------------------------------- #
def _upsert(
    session: Session,
    model: type,
    rows: list[dict],
    conflict_cols: list[str],
    batch_size: int = 500,
) -> int:
    """Insert rows with ON CONFLICT DO UPDATE (upsert). Returns row count."""
    if not rows:
        return 0
    table = model.__table__
    update_cols = [
        c.name for c in table.columns if c.name not in conflict_cols and c.name != "id"
    ]
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        stmt = pg_insert(table).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_cols,
            set_={col: stmt.excluded[col] for col in update_cols},
        )
        session.execute(stmt)
        total += len(batch)
    session.commit()
    return total


def load_fundamentals(session: Session, raw_data: Path) -> int:
    rows = list(iter_fundamentals(raw_data))
    return _upsert(
        session,
        Fundamentals,
        rows,
        ["symbol", "report_type", "period", "fiscal_date"],
    )


def load_stock_prices(session: Session, raw_data: Path) -> int:
    rows = list(iter_stock_prices(raw_data))
    return _upsert(session, StockPrice, rows, ["symbol", "date"])


def load_article_summaries(session: Session, raw_data: Path) -> int:
    rows = list(iter_article_summaries(raw_data))
    return _upsert(session, ArticleSummary, rows, ["symbol", "url"])


def load_macro_series(session: Session, raw_data: Path) -> int:
    rows = list(iter_macro_series(raw_data))
    return _upsert(session, MacroSeries, rows, ["series_id", "date"])


# Dataset name -> loader map (used by the CLI).
LOADERS: dict[str, object] = {
    "fundamentals": load_fundamentals,
    "prices": load_stock_prices,
    "articles": load_article_summaries,
    "macro": load_macro_series,
}


def load_all(session: Session, raw_data: Path) -> dict[str, int]:
    """Load every dataset. Returns the row count per dataset."""
    return {name: loader(session, raw_data) for name, loader in LOADERS.items()}
