"""Triple Barrier labels built from prices in a real database.

The prices go into Postgres through the datalake loader and `build_labels`
reads them back with its own query, exactly as `evaluate labels` does. The
expected barrier width is computed independently with pandas on the closes in
calendar order, so a wrong ordering anywhere on the way (SQL, slicing,
reversing) fails the test.
"""

import math
from datetime import date

import pytest

from scripts.evaluation.labels import build_labels
from scripts.evaluation.samples import Sample
from src.agents import llm_client, nodes
from src.agents.state import PMOpinion
from src.agents.volatility import HISTORY_SESSIONS, HORIZON_SESSIONS, VOL_SPAN
from tests.conftest import trading_days

# 400 weekdays (80 weeks) ending on Friday 2024-12-27.
HISTORY_DAYS = trading_days(date(2023, 6, 19), 400)
FRIDAY = HISTORY_DAYS[-1]
SATURDAY = date(2024, 12, 28)
FUTURE_DAYS = trading_days(date(2024, 12, 30), 8)  # Monday 2024-12-30 on
# A volatile first stretch, then mildly irregular closes: the band depends on
# which end of the series is read as the most recent.
HISTORY = [100.0 * (1.05 if i % 2 else 1.0) for i in range(150)] + [
    100.0 * (1 + 0.01 * math.sin(i / 3) + 0.002 * (i % 7)) for i in range(250)
]
P0 = HISTORY[-1]


def _expected_sigma_1w() -> float:
    """sigma_h from docs/evaluation.md §4.1, computed with pandas on the last
    250 closes up to as_of, oldest first."""
    pd = pytest.importorskip("pandas")
    returns = pd.Series(HISTORY[-HISTORY_SESSIONS:]).pct_change().dropna()
    daily = returns.ewm(span=VOL_SPAN).std().iloc[-1]
    return daily * math.sqrt(HORIZON_SESSIONS["1W"])


SIGMA = _expected_sigma_1w()


def _with_future(*moves: float) -> list[tuple[date, float]]:
    """The history, then one close per future day at p0 * (1 + move * sigma)."""
    future = [P0 * (1 + move * SIGMA) for move in moves]
    return list(zip(HISTORY_DAYS + FUTURE_DAYS, HISTORY + future))


@pytest.fixture
def prices(load_prices):
    load_prices(
        {
            # up through the upper barrier first, then down through the lower
            "UP": _with_future(0.5, 1.5, -1.5, 0, 0, 0, 0, 0),
            # down first, then up
            "DOWN": _with_future(-0.5, -1.5, 1.5, 0, 0, 0, 0, 0),
            # inside the band for the 5 sessions of 1W, outside on the 6th
            "FLAT": _with_future(0.9, -0.9, 0.5, -0.5, 0.9, 3, 3, 3),
            # history too short to estimate the volatility
            "NEW": list(
                zip(HISTORY_DAYS[-2:] + FUTURE_DAYS, [10.0, 11.0] + [12.0] * 8)
            ),
        }
    )


def _by_ticker(rows: list[dict]) -> dict[str, dict]:
    return {row["ticker"]: row for row in rows}


def test_labels_follow_the_first_barrier_touched(db_session, prices):
    samples = [
        Sample("UP", FRIDAY, "1W"),  # as_of on a trading day
        Sample("DOWN", SATURDAY, "1W"),  # as_of on a weekend
        Sample("FLAT", FRIDAY, "1W"),
    ]
    with db_session() as session:
        labels, skipped = build_labels(session, samples, k=1.0)

    assert skipped == []
    rows = _by_ticker(labels)

    # The close on as_of (Friday, or the Friday before a Saturday as_of) is p0;
    # the path starts on the next trading day.
    for row in rows.values():
        assert row["p0"] == pytest.approx(P0)
        assert row["sigma_h"] == pytest.approx(SIGMA)
        assert row["upper"] == pytest.approx(P0 * (1 + SIGMA))
        assert row["lower"] == pytest.approx(P0 * (1 - SIGMA))

    assert (rows["UP"]["label"], rows["UP"]["touch_date"]) == (
        "BUY",
        FUTURE_DAYS[1].isoformat(),
    )
    assert (rows["DOWN"]["label"], rows["DOWN"]["touch_date"]) == (
        "SELL",
        FUTURE_DAYS[1].isoformat(),
    )
    # the jump on the 6th session is past the vertical barrier
    assert (rows["FLAT"]["label"], rows["FLAT"]["touch_date"]) == ("HOLD", None)
    assert rows["FLAT"]["ret_end"] == pytest.approx(0.9 * SIGMA)


def test_wider_barriers_turn_a_touch_into_hold(db_session, prices):
    with db_session() as session:
        labels, _ = build_labels(session, [Sample("UP", FRIDAY, "1W")], k=2.0)
    assert (labels[0]["label"], labels[0]["touch_date"]) == ("HOLD", None)


def test_samples_without_enough_prices_are_skipped_with_a_reason(db_session, prices):
    samples = [
        Sample("UP", date(2023, 1, 2), "1W"),  # before the first price
        Sample("UP", FUTURE_DAYS[4], "1W"),  # horizon runs past the last price
        Sample("NEW", SATURDAY, "1W"),  # two closes are not enough history
    ]
    with db_session() as session:
        labels, skipped = build_labels(session, samples, k=1.0)

    assert labels == []
    assert [s["reason"] for s in skipped] == [
        "no price on/before as_of",
        "horizon outruns price data",
        "too little price history",
    ]


def test_pm_is_told_the_band_the_labels_use(
    db_session, prices, debate_state, monkeypatch
):
    """The Portfolio Manager's prompt quotes sigma_h from the same prices,
    read through the agents' own query (newest first): the model is judged by
    the rule it was told (docs/evaluation.md §4.1)."""
    with db_session() as session:
        [label], _ = build_labels(session, [Sample("UP", FRIDAY, "1W")], k=1.0)

    prompts: list[str] = []

    def answer(prompt, model):
        prompts.append(prompt)
        return PMOpinion(direction="HOLD", confidence=0.5, summary="s", arguments=[])

    monkeypatch.setattr(llm_client, "generate_structured", answer)
    nodes.pm_synthesize_node(debate_state(ticker="UP", as_of_date=FRIDAY))

    [prompt] = prompts
    assert f"{label['sigma_h'] * 100:.1f}%" in prompt
