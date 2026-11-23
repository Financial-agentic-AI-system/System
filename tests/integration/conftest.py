"""Fixtures for the integration tests: a real Postgres with pgvector and a
real Redis, no stand-ins for either.

Connection settings (defaults match `docker compose up -d postgres redis`):

    TEST_DATABASE_URL  postgresql://user:password@localhost:5432/mas_test
    TEST_REDIS_URL     redis://localhost:6379/15

The database name must end in `_test`: every test empties all tables, so the
fixtures refuse to touch the application database (`mas_db`). The database is
created on first use when the user is allowed to. Redis uses a separate
logical database (15 by default) and flushes only that one.

When a service cannot be reached the tests are skipped locally and fail on CI
(the `CI` environment variable is set), so a missing service in the pipeline
never passes silently.
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
import redis
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from src.agents import nodes
from src.api.v1.endpoints import meta, predict, tickers
from src.cache import debate_cache
from src.db.base import Base
from src.db.init_db import init_db
from src.db.loaders import load_stock_prices

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://user:password@localhost:5432/mas_test"
)
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

# Every module that opens its own session through `get_session()`.
_SESSION_USERS = (nodes, predict, meta, tickers)


def _unavailable(reason: str) -> None:
    if os.getenv("CI"):
        pytest.fail(reason)
    pytest.skip(reason)


def _create_database_if_missing(url) -> None:
    admin = create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", future=True
    )
    try:
        with admin.connect() as connection:
            exists = connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            )
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def db_engine():
    url = make_url(TEST_DATABASE_URL)
    if not (url.database or "").endswith("_test"):
        pytest.fail(
            f"TEST_DATABASE_URL must name a database ending in '_test', got "
            f"{url.database!r}: the integration tests empty every table."
        )
    try:
        _create_database_if_missing(url)
    except OperationalError as exc:
        _unavailable(f"Postgres not reachable at {url.host}:{url.port}: {exc}")
    engine = create_engine(url, future=True)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine, monkeypatch):
    """Session factory on the empty test database.

    Production code calls `get_session()`; it is pointed at the same factory,
    so the API, the agents and the test all read and write one database.
    """
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with db_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    for module in _SESSION_USERS:
        monkeypatch.setattr(module, "get_session", factory)
    return factory


@pytest.fixture
def redis_db(monkeypatch):
    """The debate cache on a real, empty Redis database."""
    client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError as exc:
        _unavailable(f"Redis not reachable at {TEST_REDIS_URL}: {exc}")
    client.flushdb()
    monkeypatch.setattr(debate_cache, "redis_client", client)
    yield client
    client.flushdb()
    client.close()


def trading_days(start: date, count: int) -> list[date]:
    """`count` weekdays from `start` on (holidays are ignored)."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


@pytest.fixture
def load_prices(db_session, tmp_path: Path):
    """Write daily closes as a datalake file and load it with the real loader
    (`src/db/loaders.py`), the same path `python -m src.db.load` takes.

    Takes `{ticker: [(date, close), ...]}`.
    """

    def load(series: dict[str, list[tuple[date, float]]]) -> None:
        directory = tmp_path / "raw_data" / "stock_price"
        directory.mkdir(parents=True, exist_ok=True)
        for ticker, rows in series.items():
            payload = {
                "symbol": ticker,
                "period": "daily",
                "reports": [
                    {"date": day.isoformat(), "close": str(close)}
                    for day, close in rows
                ],
            }
            (directory / f"{ticker}_daily.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        with db_session() as session:
            load_stock_prices(session, tmp_path / "raw_data")

    return load
