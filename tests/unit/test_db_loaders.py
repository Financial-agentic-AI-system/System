"""Unit tests for the datalake loading layer (no database)."""

import datetime as dt
import json
from pathlib import Path

import pytest

from src.db import loaders


# --------------------------------------------------------------------------- #
# clean_value / clean_report
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("None", None),
        ("", None),
        (".", None),
        ("379297000000", 379297000000),
        ("-211000000", -211000000),
        ("0.429075", 0.429075),
        ("USD", "USD"),
        ("2025-12-31", "2025-12-31"),
        (0.376026, 0.376026),
        (None, None),
    ],
)
def test_clean_value(raw, expected):
    assert loaders.clean_value(raw) == expected


def test_clean_value_types():
    assert isinstance(loaders.clean_value("100"), int)
    assert isinstance(loaders.clean_value("1.5"), float)


def test_clean_report_normalizes_none_and_numbers():
    report = {
        "fiscalDateEnding": "2025-12-31",
        "reportedCurrency": "USD",
        "totalAssets": "379297000000",
        "goodwill": "None",
    }
    cleaned = loaders.clean_report(report)
    assert cleaned == {
        "fiscalDateEnding": "2025-12-31",
        "reportedCurrency": "USD",
        "totalAssets": 379297000000,
        "goodwill": None,
    }


# --------------------------------------------------------------------------- #
# date parsing
# --------------------------------------------------------------------------- #
def test_parse_date():
    assert loaders.parse_date("2010-01-04") == dt.date(2010, 1, 4)
    assert loaders.parse_date("None") is None
    assert loaders.parse_date(None) is None


def test_parse_time_published():
    assert loaders.parse_time_published("20250101T093000") == dt.datetime(
        2025, 1, 1, 9, 30, 0
    )
    assert loaders.parse_time_published("bad") is None


# --------------------------------------------------------------------------- #
# iterators: file -> rows
# --------------------------------------------------------------------------- #
@pytest.fixture
def raw_data(tmp_path: Path) -> Path:
    root = tmp_path / "raw_data"

    bs = root / "BALANCE_SHEET"
    bs.mkdir(parents=True)
    (bs / "AAPL_quarterly.json").write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "period": "quarterly",
                "reports": [
                    {
                        "fiscalDateEnding": "2025-12-31",
                        "totalAssets": "379297000000",
                        "goodwill": "None",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sp = root / "stock_price"
    sp.mkdir()
    (sp / "AAPL_daily.json").write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "period": "daily",
                "reports": [
                    {"date": "2010-01-04", "close": "6.4123"},
                    {"date": "bad", "close": "1.0"},
                ],
            }
        ),
        encoding="utf-8",
    )

    art = root / "ARTICLE_SUMMARIES"
    art.mkdir()
    (art / "AAPL.json").write_text(
        json.dumps(
            [
                {
                    "title": "T",
                    "url": "http://x/1",
                    "time_published": "20250101T093000",
                    "overall_sentiment_score": 0.37,
                    "ticker_sentiment_score": "0.42",
                },
                {"title": "no url"},
            ]
        ),
        encoding="utf-8",
    )

    macro = root / "FRED_MACRO"
    macro.mkdir()
    (macro / "GDP.csv").write_text(
        "DATE,GDP\n2016-07-01,18892.639\n2016-10-01,.\n", encoding="utf-8"
    )

    return root


def test_iter_fundamentals(raw_data: Path):
    rows = list(loaders.iter_fundamentals(raw_data))
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "AAPL"
    assert row["report_type"] == "BALANCE_SHEET"
    assert row["period"] == "quarterly"
    assert row["fiscal_date"] == dt.date(2025, 12, 31)
    assert row["data"]["totalAssets"] == 379297000000
    assert row["data"]["goodwill"] is None


def test_iter_stock_prices_skips_bad_rows(raw_data: Path):
    rows = list(loaders.iter_stock_prices(raw_data))
    assert len(rows) == 1
    assert rows[0] == {
        "symbol": "AAPL",
        "date": dt.date(2010, 1, 4),
        "close": 6.4123,
    }


def test_iter_article_summaries_requires_url(raw_data: Path):
    rows = list(loaders.iter_article_summaries(raw_data))
    assert len(rows) == 1
    row = rows[0]
    assert row["url"] == "http://x/1"
    assert row["time_published"] == dt.datetime(2025, 1, 1, 9, 30, 0)
    assert row["ticker_sentiment_score"] == pytest.approx(0.42)


def test_iter_macro_series_skips_missing(raw_data: Path):
    rows = list(loaders.iter_macro_series(raw_data))
    assert rows == [
        {"series_id": "GDP", "date": dt.date(2016, 7, 1), "value": 18892.639}
    ]


def test_resolve_raw_data_explicit(raw_data: Path):
    # a directory ending in "raw_data" -> returned directly
    assert loaders.resolve_raw_data(str(raw_data)) == raw_data
    # the parent directory (datalake) -> "raw_data" appended
    assert loaders.resolve_raw_data(str(raw_data.parent)) == raw_data
