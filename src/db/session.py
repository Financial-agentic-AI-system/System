"""SQLAlchemy engine and session factory.

`DATABASE_URL` matches the `postgres` service from docker-compose. The default
points at `localhost` so the scripts also work when run locally (outside the
container).
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/mas_db",
)

engine = create_engine(DATABASE_URL, future=True)

SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def get_session() -> Session:
    """Return a new session. The caller is responsible for closing it."""
    return SessionLocal()
