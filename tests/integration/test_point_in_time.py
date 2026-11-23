"""Point-in-time reads (src/db/queries.py) on a real database.

The agents may only see what was public on `as_of_date` (docs/evaluation.md
§3.1). Every test loads a small datalake with the real loaders and checks the
boundary day by day: the last day a row is invisible and the first day it is
visible.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select

from src.db import queries
from src.db.article_embeddings import embed_article_summaries
from src.db.loaders import load_all
from src.db.models import ArticleSummary
from src.retriever.embeddings import EMBEDDING_DIM

TICKER = "ACME"
REPORT_TYPES = ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")

# (fiscal date, reported date); "None" is how Alpha Vantage writes a missing
# value, and such a report counts as published 90 days after its fiscal date.
FUNDAMENTALS = [
    ("2020-12-31", "None"),  # public from 2021-03-31 (fallback)
    ("2024-09-30", "2024-10-31"),
    ("2024-12-31", "2025-01-30"),
    ("2025-03-31", "2025-05-15"),
    ("2025-06-30", "2025-07-31"),
]
PRICES = ["2025-05-28", "2025-05-29", "2025-05-30", "2025-06-02", "2025-06-03"]
MACRO = ["2025-05-01", "2025-06-01", "2025-07-01"]
# Published around the end of Monday 2025-06-02.
ARTICLES = {
    "sunday-late": "20250601T235900",
    "monday-midnight": "20250602T000000",
    "monday-late": "20250602T235959",
    "tuesday-midnight": "20250603T000000",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _article(name: str, published: str) -> dict:
    return {
        "title": f"ACME {name}",
        "url": f"https://news.example/{name}",
        "time_published": published,
        "summary": f"Article published {published}.",
    }


@pytest.fixture
def session(db_session, tmp_path):
    raw = tmp_path / "raw_data"
    for report_type in REPORT_TYPES:
        _write_json(
            raw / report_type / f"{TICKER}_quarterly.json",
            {
                "symbol": TICKER,
                "period": "quarterly",
                "reports": [
                    {"fiscalDateEnding": fiscal, "reportedDate": reported, "x": "1"}
                    for fiscal, reported in FUNDAMENTALS
                ],
            },
        )
    _write_json(
        raw / "stock_price" / f"{TICKER}_daily.json",
        {
            "symbol": TICKER,
            "reports": [{"date": day, "close": "10.0"} for day in PRICES],
        },
    )
    macro = raw / "FRED_MACRO" / "FEDFUNDS.csv"
    macro.parent.mkdir(parents=True)
    macro.write_text(
        "DATE,FEDFUNDS\n" + "".join(f"{day},4.33\n" for day in MACRO),
        encoding="utf-8",
    )
    _write_json(
        raw / "ARTICLE_SUMMARIES" / f"{TICKER}.json",
        [_article(name, published) for name, published in ARTICLES.items()],
    )
    # the same day, another ticker: never visible for ACME
    _write_json(
        raw / "ARTICLE_SUMMARIES" / "OTHER.json",
        [_article("other-ticker", "20250602T120000")],
    )

    with db_session() as session:
        load_all(session, raw)
        yield session


def _fiscal_dates(session, as_of: date, **kwargs) -> list[str]:
    rows = queries.fetch_fundamentals(session, TICKER, as_of, **kwargs)
    return sorted({row.fiscal_date.isoformat() for row in rows})


def _names(articles) -> list[str]:
    return [article.url.rsplit("/", 1)[1] for article in articles]


# --------------------------------------------------------------------------- #
# fundamentals: published = reported_date, else fiscal_date + 90 days
# --------------------------------------------------------------------------- #
def test_report_is_visible_from_its_reported_date(session):
    # Q1 2025 ended on 2025-03-31 but was published on 2025-05-15
    assert "2025-03-31" not in _fiscal_dates(session, date(2025, 5, 14))
    assert "2025-03-31" in _fiscal_dates(session, date(2025, 5, 15))


def test_report_without_reported_date_is_visible_90_days_after_fiscal_date(session):
    assert _fiscal_dates(session, date(2021, 3, 30)) == []
    assert _fiscal_dates(session, date(2021, 3, 31)) == ["2020-12-31"]


def test_fundamentals_as_of_a_date(session):
    """Q2 2025 (fiscal 2025-06-30) is not public yet on 2025-06-02; every
    visible quarter comes with all three statements."""
    as_of = date(2025, 6, 2)
    assert _fiscal_dates(session, as_of) == [
        "2020-12-31",
        "2024-09-30",
        "2024-12-31",
        "2025-03-31",
    ]
    rows = queries.fetch_fundamentals(session, TICKER, as_of)
    assert len(rows) == 4 * len(REPORT_TYPES)
    # `quarters` keeps the newest *published* quarters
    assert _fiscal_dates(session, as_of, quarters=2) == ["2024-12-31", "2025-03-31"]


# --------------------------------------------------------------------------- #
# prices and macro: date <= as_of_date
# --------------------------------------------------------------------------- #
def test_prices_up_to_and_including_as_of(session):
    rows = queries.fetch_prices(session, TICKER, date(2025, 6, 2), lookback_days=4)
    # newest first; the window starts on as_of - 4 days (2025-05-29), inclusive
    assert [row.date.isoformat() for row in rows] == [
        "2025-06-02",
        "2025-05-30",
        "2025-05-29",
    ]


def test_macro_up_to_and_including_as_of(session):
    """Filtered by the observation date. FRED publishes a month's value weeks
    later; that lag is not modelled (docs/evaluation.md §3.1, known gap)."""
    rows = queries.fetch_macro_series(session, date(2025, 6, 1))
    assert [row.date.isoformat() for row in rows] == ["2025-06-01", "2025-05-01"]


# --------------------------------------------------------------------------- #
# articles: time_published < as_of_date + 1 day
# --------------------------------------------------------------------------- #
VISIBLE_ON_MONDAY = ["monday-late", "monday-midnight", "sunday-late"]  # newest first
MONDAY = date(2025, 6, 2)


def test_recent_articles_include_the_whole_as_of_day(session):
    articles = queries.fetch_recent_articles(session, TICKER, MONDAY)
    assert _names(articles) == VISIBLE_ON_MONDAY


def test_articles_found_earlier_are_filtered_again(session):
    """Ids kept in the debate state are re-read with the same bound: an id of a
    later article, or of another ticker, returns nothing."""
    all_ids = list(session.scalars(select(ArticleSummary.id)))
    articles = queries.fetch_articles_by_ids(session, TICKER, MONDAY, all_ids)
    assert _names(articles) == VISIBLE_ON_MONDAY


def test_semantic_search_never_returns_a_later_article(session):
    """Every vector is identical, so every article is a perfect match: only
    the date and ticker filters decide what comes back."""
    vector = [1.0] + [0.0] * (EMBEDDING_DIM - 1)

    # before embedding: the visible articles are reported as missing vectors
    assert queries.count_articles_without_embedding(session, TICKER, MONDAY) == 3
    embed_article_summaries(session, embed=lambda texts: [vector for _ in texts])
    assert queries.count_articles_without_embedding(session, TICKER, MONDAY) == 0

    for variant in ("summary", "title_summary"):
        hits = queries.fetch_similar_articles(
            session, TICKER, MONDAY, vector, variant=variant
        )
        assert sorted(_names(a for a, _ in hits)) == sorted(VISIBLE_ON_MONDAY)
        assert all(similarity == pytest.approx(1.0) for _, similarity in hits)
