"""Semantic search over `article_summaries`: embed the question, then run the
point-in-time similarity query in `src/db/queries.py`.
"""

from datetime import date

from sqlalchemy.orm import Session

from src.db.models import ArticleSummary
from src.db.queries import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_VARIANT,
    EmbeddingVariant,
    fetch_similar_articles,
)
from src.retriever import embeddings


def search_articles(
    session: Session,
    ticker: str,
    as_of_date: date,
    query: str,
    limit: int = 10,
    variant: EmbeddingVariant = DEFAULT_VARIANT,
    min_similarity: float | None = DEFAULT_MIN_SIMILARITY,
) -> list[tuple[ArticleSummary, float]]:
    """(article, similarity) pairs most relevant to `query`, published on or
    before `as_of_date`.
    """
    return fetch_similar_articles(
        session,
        ticker,
        as_of_date,
        embeddings.embed_query(query),
        limit,
        variant,
        min_similarity,
    )
