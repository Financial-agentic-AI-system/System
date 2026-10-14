"""Create the database schema.

Run:
    python -m src.db.init_db
"""

from sqlalchemy import text

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
)


def init_db() -> None:
    """Create the pgvector extension, all tables, and any later-added columns."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        for statement in _ADDED_COLUMNS:
            connection.execute(text(statement))


if __name__ == "__main__":
    init_db()
    print("Tables created (or already existed).")
