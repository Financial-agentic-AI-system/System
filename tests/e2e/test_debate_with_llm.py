"""The debate on the real LLM, over a real database and Redis.

Nothing is replaced: a small datalake is written to disk and loaded with the
real loaders, the articles are embedded with the real embedding API (when the
Sentiment agent's search is enabled), and every agent calls the real model.

These tests cost tokens and need credentials, so they live outside the
directories CI runs. Run them explicitly:

    docker compose up -d postgres redis
    gcloud auth application-default login   # once
    uv run --env-file .env pytest tests/e2e

Without `GCP_PROJECT_ID` (or `GEMINI_API_KEY`) they are skipped.

The model's answers are not deterministic, so the tests check the shape and
the invariants of a run (valid values, the order of the protocol, no article
from after `as_of_date`), never the content of an opinion.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from src.agents import llm_client, nodes, usage
from src.agents.state import MAX_ROUNDS, Direction
from src.cache import debate_cache
from src.db.article_embeddings import embed_article_summaries
from src.db.loaders import load_all
from src.worker import tasks
from tests.conftest import trading_days

pytestmark = pytest.mark.skipif(
    not (llm_client._PROJECT_ID or llm_client._API_KEY),
    reason="no LLM credentials: run with `uv run --env-file .env pytest tests/e2e`",
)

TICKER = "ACME"  # fictional, so the model cannot know how it went on
AS_OF = date(2025, 6, 2)
LATE_ARTICLE_URL = "https://news.example/acme-after-as-of"

# fiscal date, reported date; the last report is published after AS_OF
QUARTERS = [
    ("2024-06-30", "2024-08-01"),
    ("2024-09-30", "2024-10-31"),
    ("2024-12-31", "2025-01-30"),
    ("2025-03-31", "2025-05-01"),
    ("2025-06-30", "2025-07-31"),
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reports(build) -> dict:
    return {
        "symbol": TICKER,
        "period": "quarterly",
        "reports": [
            {"fiscalDateEnding": fiscal, "reportedDate": reported, **build(i)}
            for i, (fiscal, reported) in enumerate(QUARTERS)
        ],
    }


def _article(day: date, title: str, summary: str, label: str, score: float) -> dict:
    return {
        "title": title,
        "url": f"https://news.example/acme-{day.isoformat()}",
        "time_published": day.strftime("%Y%m%dT093000"),
        "source": "Example News",
        "summary": summary,
        "overall_sentiment_score": score,
        "overall_sentiment_label": label,
        "ticker_sentiment_score": str(score),
        "ticker_sentiment_label": label,
        "relevance_score": "0.9",
    }


def _write_datalake(raw: Path) -> None:
    """A fictional company in the datalake's own file formats."""
    _write_json(
        raw / "INCOME_STATEMENT" / f"{TICKER}_quarterly.json",
        _reports(
            lambda i: {
                "reportedCurrency": "USD",
                "totalRevenue": str(9_000_000_000 + 400_000_000 * i),
                "grossProfit": str(3_800_000_000 + 200_000_000 * i),
                "netIncome": str(1_100_000_000 + 90_000_000 * i),
            }
        ),
    )
    _write_json(
        raw / "BALANCE_SHEET" / f"{TICKER}_quarterly.json",
        _reports(
            lambda i: {
                "reportedCurrency": "USD",
                "totalAssets": str(52_000_000_000 + 1_000_000_000 * i),
                "totalLiabilities": str(31_000_000_000 + 300_000_000 * i),
                "cashAndCashEquivalentsAtCarryingValue": str(6_000_000_000),
            }
        ),
    )
    _write_json(
        raw / "CASH_FLOW" / f"{TICKER}_quarterly.json",
        _reports(
            lambda i: {
                "reportedCurrency": "USD",
                "operatingCashflow": str(1_600_000_000 + 80_000_000 * i),
                "capitalExpenditures": str(-400_000_000),
            }
        ),
    )

    days = trading_days(date(2024, 6, 3), 320)  # until 2025-08
    _write_json(
        raw / "stock_price" / f"{TICKER}_daily.json",
        {
            "symbol": TICKER,
            "period": "daily",
            "reports": [
                {"date": d.isoformat(), "close": f"{80 + 0.05 * i + (i % 5):.2f}"}
                for i, d in enumerate(days)
            ],
        },
    )

    months = [date(2023 + m // 12, m % 12 + 1, 1) for m in range(31)]  # to 2025-07
    for series, start, step in (
        ("FEDFUNDS", 5.3, -0.03),
        ("CPIAUCSL", 300.0, 0.6),
        ("UNRATE", 3.6, 0.02),
    ):
        lines = ["DATE," + series] + [
            f"{d.isoformat()},{start + step * i:.2f}" for i, d in enumerate(months)
        ]
        path = raw / "FRED_MACRO" / f"{series}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    articles = [
        _article(
            date(2025, 5, 1),
            "ACME beats quarterly revenue estimates",
            "ACME reported revenue up 4% on strong demand for its industrial "
            "sensors and kept its full-year guidance.",
            "Somewhat-Bullish",
            0.3,
        ),
        _article(
            date(2025, 5, 14),
            "ACME faces supplier delays in Asia",
            "Analysts warn that component shortages could slow ACME shipments "
            "during the summer quarter.",
            "Somewhat-Bearish",
            -0.2,
        ),
        _article(
            date(2025, 5, 27),
            "ACME announces share buyback",
            "The board approved a 2 billion dollar buyback programme over the "
            "next two years.",
            "Bullish",
            0.4,
        ),
        _article(
            date(2025, 6, 2),
            "ACME CEO comments on tariffs",
            "The chief executive said new tariffs would have a limited effect on "
            "margins this year.",
            "Neutral",
            0.05,
        ),
    ]
    late = _article(
        date(2025, 6, 20),
        "ACME cuts guidance after plant fire",
        "A fire at the main ACME plant halts production; the company cuts its "
        "full-year guidance sharply.",
        "Bearish",
        -0.6,
    )
    late["url"] = LATE_ARTICLE_URL
    _write_json(raw / "ARTICLE_SUMMARIES" / f"{TICKER}.json", [*articles, late])


@pytest.fixture
def datalake(db_session, tmp_path):
    raw = tmp_path / "raw_data"
    _write_datalake(raw)
    with db_session() as session:
        load_all(session, raw)
        if nodes.retrieval_enabled():
            embed_article_summaries(session)


def test_financial_agent_answers_from_the_database(datalake, debate_state):
    state = debate_state(ticker=TICKER, as_of_date=AS_OF)

    with usage.track() as calls:
        report = nodes.financial_agent_node(state)["financial_report"]

    assert report.opinion.strip()
    assert report.reasoning.strip()
    assert (calls.calls, calls.failed_calls) == (1, 0)
    assert calls.input_tokens > 0 and calls.output_tokens > 0


def test_full_debate_runs_on_the_real_llm(datalake, redis_db):
    result = tasks.execute_debate(TICKER, "1M", AS_OF, task_id="llm-smoke")

    assert result.direction in get_args(Direction)
    assert 0.0 <= result.confidence <= 1.0
    assert 1 <= result.rounds_used <= MAX_ROUNDS
    for report in (
        result.agent_reports.financial,
        result.agent_reports.sentiment,
        result.agent_reports.macro,
    ):
        assert report.opinion.strip() and report.reasoning.strip()

    status = debate_cache.get_status("llm-smoke")
    assert (status.status, status.round_number) == ("DONE", result.rounds_used)
    assert debate_cache.get_result("llm-smoke") == result

    history = debate_cache.get_history("llm-smoke")
    assert {h.node for h in history[:3]} == {"financial", "sentiment", "macro"}
    assert history[-1].node == "finalize"
    assert [h.node for h in history].count("critic") == result.rounds_used

    # Whatever the Sentiment agent searched for, it only found articles
    # published by the end of AS_OF (docs/evaluation.md §3.1).
    sources = [
        source
        for entry in history
        for source in entry.delta.get("sentiment_sources", [])
    ]
    cutoff = datetime.combine(AS_OF + timedelta(days=1), datetime.min.time())
    for source in sources:
        assert source["url"] != LATE_ARTICLE_URL
        assert source["published"] is not None
        assert datetime.fromisoformat(source["published"]) < cutoff
