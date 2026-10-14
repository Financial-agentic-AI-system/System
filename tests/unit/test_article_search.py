"""Unit tests for the article queries (src/db/queries.py) and the retriever
wrapper (src/retriever/search.py) — no database, no embedding API."""

from datetime import date

import pytest
from sqlalchemy.dialects import postgresql

from src.db import queries
from src.retriever import search
from src.retriever.embeddings import EMBEDDING_DIM

VECTOR = [0.1] * EMBEDDING_DIM


class _RecordingSession:
    def __init__(self, rows=None) -> None:
        self.statement = None
        self._rows = rows or []

    def execute(self, statement):
        self.statement = statement
        return iter(self._rows)

    def scalars(self, statement):
        self.statement = statement
        return iter(self._rows)


def _sql(session: _RecordingSession) -> str:
    return str(
        session.statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def _selected_columns(sql: str) -> str:
    return sql.split("FROM")[0]


@pytest.mark.parametrize(
    ("variant", "column"),
    [
        ("summary", "article_summaries.summary_embedding"),
        ("title_summary", "article_summaries.title_summary_embedding"),
    ],
)
def test_similarity_query_shape(variant, column):
    session = _RecordingSession(rows=[("article", 0.83)])

    rows = queries.fetch_similar_articles(
        session,
        "AAPL",
        date(2026, 1, 15),
        VECTOR,
        limit=5,
        variant=variant,
        min_similarity=None,
    )

    assert rows == [("article", 0.83)]
    sql = _sql(session)
    where, order = sql.split("WHERE")[1].split("ORDER BY")
    assert "article_summaries.symbol = 'AAPL'" in where
    # day-inclusive point-in-time bound: strictly before the next midnight
    assert "article_summaries.time_published < '2026-01-16'" in where
    assert f"{column} IS NOT NULL" in where
    assert f"{column} <=> " in order  # pgvector cosine distance
    assert "LIMIT 5" in order
    # similarity is returned, the vectors themselves are not
    assert "AS similarity" in _selected_columns(sql)
    assert "_embedding," not in _selected_columns(sql)


def test_similarity_threshold_filters_by_distance():
    session = _RecordingSession()

    queries.fetch_similar_articles(
        session, "AAPL", date(2026, 1, 15), VECTOR, min_similarity=0.6
    )

    where = _sql(session).split("WHERE")[1].split("ORDER BY")[0]
    assert "<= 0.4" in where  # similarity >= 0.6  <=>  distance <= 0.4


def test_defaults_use_title_summary_and_threshold():
    session = _RecordingSession()

    queries.fetch_similar_articles(session, "AAPL", date(2026, 1, 15), VECTOR)

    where = _sql(session).split("WHERE")[1].split("ORDER BY")[0]
    assert "article_summaries.title_summary_embedding <=> " in where
    assert f"<= {1 - queries.DEFAULT_MIN_SIMILARITY}" in where


def test_recent_articles_do_not_load_embeddings():
    session = _RecordingSession()

    queries.fetch_recent_articles(session, "AAPL", date(2026, 1, 15))

    assert "embedding" not in _selected_columns(_sql(session))


def test_search_articles_embeds_the_query_as_a_query(monkeypatch):
    seen = {}

    def fake_embed_query(text):
        seen["text"] = text
        return VECTOR

    def fake_fetch(session, ticker, as_of_date, embedding, limit, variant, min_sim):
        seen.update(
            ticker=ticker,
            embedding=embedding,
            limit=limit,
            variant=variant,
            min_similarity=min_sim,
        )
        return [("article", 0.9)]

    monkeypatch.setattr(search.embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(search, "fetch_similar_articles", fake_fetch)

    rows = search.search_articles(
        None,
        "AAPL",
        date(2026, 1, 15),
        "guidance cut",
        limit=3,
        variant="title_summary",
        min_similarity=0.5,
    )

    assert rows == [("article", 0.9)]
    assert seen == {
        "text": "guidance cut",
        "ticker": "AAPL",
        "embedding": VECTOR,
        "limit": 3,
        "variant": "title_summary",
        "min_similarity": 0.5,
    }
