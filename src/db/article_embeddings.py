"""Fill `summary_embedding` and `title_summary_embedding` of
`article_summaries`. Called by `python -m src.db.load`; only rows whose
embedding is still NULL are sent to the model, so re-runs resume.
"""

from collections.abc import Callable, Sequence

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from src.db.models import ArticleSummary
from src.retriever import embeddings

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_ROWS_PER_COMMIT = 200


def summary_text(summary: str | None) -> str | None:
    text = (summary or "").strip()
    return text or None


def title_summary_text(title: str | None, summary: str | None) -> str | None:
    """Title and summary joined; whichever exists if the other is missing."""
    parts = [part.strip() for part in (title, summary) if part and part.strip()]
    return "\n\n".join(parts) or None


def _embed_column(
    session: Session,
    column: str,
    texts_by_id: dict[int, str],
    embed: EmbedFn,
) -> int:
    ids = list(texts_by_id)
    if ids:
        print(f"    {column}: {len(ids)} to embed", flush=True)
    for start in range(0, len(ids), _ROWS_PER_COMMIT):
        chunk = ids[start : start + _ROWS_PER_COMMIT]
        vectors = embed([texts_by_id[row_id] for row_id in chunk])
        session.execute(
            update(ArticleSummary),
            [{"id": row_id, column: vec} for row_id, vec in zip(chunk, vectors)],
        )
        session.commit()
        done = start + len(chunk)
        if done % (_ROWS_PER_COMMIT * 10) == 0 or done == len(ids):
            print(f"    {column}: {done}/{len(ids)}", flush=True)
    return len(ids)


def embed_article_summaries(
    session: Session, embed: EmbedFn = embeddings.embed_texts
) -> dict[str, int]:
    """Embed articles missing a vector. Returns rows embedded per column."""
    rows = session.execute(
        select(
            ArticleSummary.id,
            ArticleSummary.title,
            ArticleSummary.summary,
            ArticleSummary.summary_embedding.is_(None),
            ArticleSummary.title_summary_embedding.is_(None),
        ).where(
            or_(
                ArticleSummary.summary_embedding.is_(None),
                ArticleSummary.title_summary_embedding.is_(None),
            )
        )
    ).all()

    summary_texts: dict[int, str] = {}
    title_summary_texts: dict[int, str] = {}
    for row_id, title, summary, needs_summary, needs_title_summary in rows:
        if needs_summary and (text := summary_text(summary)):
            summary_texts[row_id] = text
        if needs_title_summary and (text := title_summary_text(title, summary)):
            title_summary_texts[row_id] = text

    return {
        "summary_embedding": _embed_column(
            session, "summary_embedding", summary_texts, embed
        ),
        "title_summary_embedding": _embed_column(
            session, "title_summary_embedding", title_summary_texts, embed
        ),
    }
