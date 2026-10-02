"""Unit tests for the similarity query (src/db/queries.py) and its retriever
wrapper (src/retriever/search.py) — no database, no embedding API."""

from datetime import date

import pytest
from sqlalchemy.dialects import postgresql

from src.db import queries
from src.retriever import search
from src.retriever.embeddings import EMBEDDING_DIM

VECTOR = [0.1] * EMBEDDING_DIM


class _RecordingSession:
    def __init__(self) -> None:
        self.statement = None

    def scalars(self, statement):
        self.statement = statement
        return ["row"]


def _sql(session: _RecordingSession) -> str:
    return str(
        session.statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


@pytest.mark.parametrize(
    ("variant", "column"),
    [
        ("summary", "article_summaries.summary_embedding"),
        ("title_summary", "article_summaries.title_summary_embedding"),
    ],
)
def test_similarity_query_shape(variant, column):
    session = _RecordingSession()

    rows = queries.fetch_similar_articles(
        session, "AAPL", date(2026, 1, 15), VECTOR, limit=5, variant=variant
    )

    assert rows == ["row"]
    sql = _sql(session)
    where, order = sql.split("WHERE")[1].split("ORDER BY")
    assert "article_summaries.symbol = 'AAPL'" in where
    # day-inclusive point-in-time bound: strictly before the next midnight
    assert "article_summaries.time_published < '2026-01-16'" in where
    assert f"{column} IS NOT NULL" in where
    assert f"{column} <=> " in order  # pgvector cosine distance
    assert "LIMIT 5" in order
    # the 768-float vectors are not shipped back with every row
    assert "embedding" not in sql.split("FROM")[0]


def test_search_articles_embeds_the_query_as_a_query(monkeypatch):
    seen = {}

    def fake_embed_query(text):
        seen["text"] = text
        return VECTOR

    def fake_fetch(session, ticker, as_of_date, embedding, limit, variant):
        seen.update(ticker=ticker, embedding=embedding, limit=limit, variant=variant)
        return ["row"]

    monkeypatch.setattr(search.embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(search, "fetch_similar_articles", fake_fetch)

    rows = search.search_articles(
        None,
        "AAPL",
        date(2026, 1, 15),
        "guidance cut",
        limit=3,
        variant="title_summary",
    )

    assert rows == ["row"]
    assert seen == {
        "text": "guidance cut",
        "ticker": "AAPL",
        "embedding": VECTOR,
        "limit": 3,
        "variant": "title_summary",
    }
