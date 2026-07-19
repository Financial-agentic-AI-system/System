"""Create the database schema.

Run:
    python -m src.db.init_db
"""

from src.db import models  # noqa: F401  (registers tables in metadata)
from src.db.base import Base
from src.db.session import engine


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Tables created (or already existed).")
