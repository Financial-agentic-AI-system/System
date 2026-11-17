"""Create the database schema.

Run:
    python -m src.db.init_db
"""

from sqlalchemy import Engine, text

from src.db import models  # noqa: F401  (registers tables in metadata)
from src.db.base import Base
from src.db.session import engine
from src.retriever.embeddings import EMBEDDING_DIM

# `create_all` never alters an existing table.
_ADDED_COLUMNS: tuple[str, ...] = (
    "ALTER TABLE article_summaries "
    f"ADD COLUMN IF NOT EXISTS summary_embedding vector({EMBEDDING_DIM})",
    "ALTER TABLE article_summaries "
    f"ADD COLUMN IF NOT EXISTS title_summary_embedding vector({EMBEDDING_DIM})",
    "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS reported_date DATE",
    "CREATE INDEX IF NOT EXISTS ix_fundamentals_reported_date "
    "ON fundamentals (reported_date)",
)


def init_db(bind: Engine = engine) -> None:
    """Create the pgvector extension, all tables, and any later-added columns.

    `bind` defaults to the application database; the integration tests pass
    their own engine.
    """
    with bind.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=bind)
    with bind.begin() as connection:
        for statement in _ADDED_COLUMNS:
            connection.execute(text(statement))


if __name__ == "__main__":
    init_db()
    print("Tables created (or already existed).")
