"""Unit tests for src/db/article_embeddings.py and the loader's handling of
the embedding columns — no database, no embedding API."""

from sqlalchemy.dialects import postgresql

from src.db import article_embeddings, loaders
from src.db.models import ArticleSummary


class FakeSession:
    """Returns canned rows for the SELECT, records the bulk UPDATEs."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.updates: list[list[dict]] = []
        self.commits = 0

    def execute(self, statement, params=None):
        if params is None:
            return self
        self.updates.append(params)
        return None

    def all(self) -> list[tuple]:
        return self._rows

    def commit(self) -> None:
        self.commits += 1


def _fake_embed(seen: list[str]):
    def embed(texts):
        seen.extend(texts)
        return [[float(len(text))] for text in texts]

    return embed


def test_text_builders():
    assert article_embeddings.summary_text("  s ") == "s"
    assert article_embeddings.summary_text("") is None
    assert article_embeddings.summary_text(None) is None
    assert article_embeddings.title_summary_text("T", "S") == "T\n\nS"
    assert article_embeddings.title_summary_text("T", None) == "T"
    assert article_embeddings.title_summary_text(None, " ") is None


def test_embeds_only_missing_columns():
    # (id, title, summary, summary_embedding IS NULL, title_summary_embedding IS NULL)
    session = FakeSession(
        [
            (1, "T1", "S1", True, True),
            (2, "T2", "S2", False, True),  # summary vector already there
            (3, "T3", None, True, True),  # no summary: title-only variant
        ]
    )
    seen: list[str] = []

    counts = article_embeddings.embed_article_summaries(session, _fake_embed(seen))

    assert counts == {"summary_embedding": 1, "title_summary_embedding": 3}
    assert seen == ["S1", "T1\n\nS1", "T2\n\nS2", "T3"]
    assert session.updates == [
        [{"id": 1, "summary_embedding": [2.0]}],
        [
            {"id": 1, "title_summary_embedding": [6.0]},
            {"id": 2, "title_summary_embedding": [6.0]},
            {"id": 3, "title_summary_embedding": [2.0]},
        ],
    ]
    assert session.commits == 2


def test_nothing_to_embed_makes_no_calls():
    session = FakeSession([])
    seen: list[str] = []
    counts = article_embeddings.embed_article_summaries(session, _fake_embed(seen))
    assert counts == {"summary_embedding": 0, "title_summary_embedding": 0}
    assert seen == [] and session.updates == []


def test_commits_per_chunk(monkeypatch):
    monkeypatch.setattr(article_embeddings, "_ROWS_PER_COMMIT", 2)
    session = FakeSession([(i, None, f"s{i}", True, False) for i in range(5)])
    article_embeddings.embed_article_summaries(session, _fake_embed([]))
    assert [len(batch) for batch in session.updates] == [2, 2, 1]
    assert session.commits == 3


class _RecordingSession:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)

    def commit(self) -> None:
        pass


def _article_row() -> dict:
    row = dict.fromkeys(
        c.name for c in ArticleSummary.__table__.columns if "embedding" not in c.name
    )
    row.pop("id")
    row.update(symbol="AAPL", url="u", title="T", summary="S")
    return row


def test_article_upsert_keeps_embeddings_unless_text_changed():
    """Re-loading the datalake must not wipe vectors: the embedding columns are
    only reset (to NULL) when the title/summary they were built from changed."""
    session = _RecordingSession()
    loaders._upsert(
        session,
        ArticleSummary,
        [_article_row()],
        ["symbol", "url"],
        stale_on_change=loaders._reset_stale_embeddings,
    )
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    set_clause = sql.split("DO UPDATE SET")[1]

    assert "summary_embedding = excluded.summary_embedding" not in set_clause
    assert (
        "summary_embedding = CASE WHEN (excluded.summary IS DISTINCT FROM "
        "article_summaries.summary)" in set_clause
    )
    assert "title_summary_embedding = CASE WHEN" in set_clause
    assert "excluded.title IS DISTINCT FROM article_summaries.title" in set_clause


def test_upsert_without_hook_only_updates_row_columns():
    session = _RecordingSession()
    loaders._upsert(session, ArticleSummary, [_article_row()], ["symbol", "url"])
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "embedding" not in sql.split("DO UPDATE SET")[1]
