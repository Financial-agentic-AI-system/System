"""Re-loading the datalake into a real database (`src/db/loaders.py`).

`python -m src.db.load` is run again whenever the datalake grows, so loading
must be an idempotent upsert: no duplicates, changed values updated, and the
article vectors kept unless the text they were built from changed (they cost
an embedding API call each).
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from src.db.article_embeddings import embed_article_summaries
from src.db.base import Base
from src.db.loaders import load_all
from src.db.models import ArticleSummary, Fundamentals, StockPrice
from src.retriever.embeddings import EMBEDDING_DIM

TICKER = "ACME"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_datalake(raw: Path, *, close: str, reported: str, articles: dict) -> None:
    _write_json(
        raw / "INCOME_STATEMENT" / f"{TICKER}_quarterly.json",
        {
            "symbol": TICKER,
            "period": "quarterly",
            "reports": [
                {"fiscalDateEnding": "2025-03-31", "reportedDate": reported, "x": "1"}
            ],
        },
    )
    _write_json(
        raw / "stock_price" / f"{TICKER}_daily.json",
        {
            "symbol": TICKER,
            "reports": [
                {"date": "2025-06-02", "close": close},
                {"date": "2025-06-03", "close": "11.0"},
            ],
        },
    )
    macro = raw / "FRED_MACRO" / "FEDFUNDS.csv"
    macro.parent.mkdir(parents=True, exist_ok=True)
    macro.write_text("DATE,FEDFUNDS\n2025-05-01,4.33\n", encoding="utf-8")
    _write_json(
        raw / "ARTICLE_SUMMARIES" / f"{TICKER}.json",
        [
            {
                "title": title,
                "url": f"https://news.example/{key}",
                "time_published": "20250602T120000",
                "summary": summary,
            }
            for key, (title, summary) in articles.items()
        ],
    )


ARTICLES = {
    "unchanged": ("Title A", "Summary A"),
    "new-title": ("Title B", "Summary B"),
    "new-summary": ("Title C", "Summary C"),
}


def _embed(texts):
    """A distinct, deterministic unit vector per text."""
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIM
        vector[sum(map(ord, text)) % EMBEDDING_DIM] = 1.0
        vectors.append(vector)
    return vectors


def _counts(session) -> dict[str, int]:
    return {
        table.name: session.scalar(select(func.count()).select_from(table))
        for table in Base.metadata.sorted_tables
    }


def _vectors(session) -> dict[str, tuple]:
    """url suffix -> (summary vector, title+summary vector), as lists or None."""
    rows = session.execute(
        select(
            ArticleSummary.url,
            ArticleSummary.summary_embedding,
            ArticleSummary.title_summary_embedding,
        )
    )
    return {
        url.rsplit("/", 1)[1]: tuple(
            None if vector is None else [float(v) for v in vector]
            for vector in (summary, title_summary)
        )
        for url, summary, title_summary in rows
    }


@pytest.fixture
def raw(tmp_path):
    return tmp_path / "raw_data"


def test_reloading_updates_rows_without_duplicates(db_session, raw):
    _write_datalake(raw, close="10.0", reported="None", articles=ARTICLES)
    with db_session() as session:
        load_all(session, raw)
        first = _counts(session)

    # the datalake is fetched again: a corrected close, a known reported date
    _write_datalake(raw, close="10.5", reported="2025-05-15", articles=ARTICLES)
    with db_session() as session:
        load_all(session, raw)

        assert _counts(session) == first
        assert first == {
            "fundamentals": 1,
            "stock_prices": 2,
            "article_summaries": 3,
            "macro_series": 1,
        }
        price = session.scalar(
            select(StockPrice.close).where(StockPrice.date == "2025-06-02")
        )
        assert float(price) == 10.5
        reported = session.scalar(select(Fundamentals.reported_date))
        assert reported.isoformat() == "2025-05-15"


def test_vectors_survive_a_reload_unless_their_text_changed(db_session, raw):
    _write_datalake(raw, close="10.0", reported="None", articles=ARTICLES)
    with db_session() as session:
        load_all(session, raw)
        assert embed_article_summaries(session, embed=_embed) == {
            "summary_embedding": 3,
            "title_summary_embedding": 3,
        }
        before = _vectors(session)

    changed = {
        **ARTICLES,
        "new-title": ("Title B, corrected", "Summary B"),
        "new-summary": ("Title C", "Summary C, corrected"),
    }
    _write_datalake(raw, close="10.0", reported="None", articles=changed)
    with db_session() as session:
        load_all(session, raw)
        after = _vectors(session)

        assert after["unchanged"] == before["unchanged"]
        # the title only feeds the title+summary vector
        assert after["new-title"] == (before["new-title"][0], None)
        # the summary feeds both
        assert after["new-summary"] == (None, None)

        # the next embedding run sends only what was reset
        assert embed_article_summaries(session, embed=_embed) == {
            "summary_embedding": 1,
            "title_summary_embedding": 2,
        }
        assert all(None not in pair for pair in _vectors(session).values())
