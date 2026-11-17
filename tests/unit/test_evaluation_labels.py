"""Unit tests for the Triple Barrier labels (scripts/evaluation/labels.py) and
the volatility that sets the barrier width (src/agents/volatility.py).

`build_labels` itself, on prices in a real database, is tested in
tests/integration/test_labels_from_database.py.
"""

import math
from datetime import date, timedelta

import pytest

from scripts.evaluation.labels import triple_barrier
from src.agents.volatility import HORIZON_SESSIONS, VOL_SPAN, horizon_sigma

# p0 * (1 +/- k * sigma_h) with exactly representable barriers: 125 and 75.
P0, SIGMA_H, K = 100.0, 0.25, 1.0
DAY = date(2025, 3, 3)


def _path(*closes: float) -> list[tuple[date, float]]:
    return [(DAY + timedelta(days=i), close) for i, close in enumerate(closes)]


@pytest.mark.parametrize(
    ("closes", "label", "touch_index"),
    [
        # the first barrier touched decides, whatever happens later
        ((110.0, 130.0, 70.0), "BUY", 1),
        ((90.0, 70.0, 130.0), "SELL", 1),
        # touching a barrier exactly counts
        ((100.0, 125.0), "BUY", 1),
        ((75.0,), "SELL", 0),
        # a touch on the last day of the horizon still counts
        ((100.0, 100.0, 100.0, 100.0, 126.0), "BUY", 4),
        # neither barrier within the horizon: the vertical barrier, HOLD
        ((124.9, 75.1, 100.0), "HOLD", None),
    ],
)
def test_triple_barrier_first_touch_decides(closes, label, touch_index):
    path = _path(*closes)
    expected_touch = None if touch_index is None else path[touch_index][0]
    assert triple_barrier(P0, path, SIGMA_H, K) == (label, expected_touch)


def test_barrier_width_scales_with_k():
    path = _path(130.0)
    assert triple_barrier(P0, path, SIGMA_H, 1.0) == ("BUY", DAY)
    # k = 2 puts the barriers at 150 / 50
    assert triple_barrier(P0, path, SIGMA_H, 2.0) == ("HOLD", None)


def test_horizon_sigma_matches_pandas_on_chronological_closes():
    """`horizon_sigma` takes closes newest first. pandas, given the same
    closes oldest first, must agree: the EWM weights the newest returns most."""
    pd = pytest.importorskip("pandas")
    closes = [
        100.0 * (1 + 0.01 * math.sin(i / 3) + 0.002 * (i % 5)) for i in range(250)
    ]

    returns = pd.Series(closes).pct_change().dropna()
    expected = returns.ewm(span=VOL_SPAN).std().iloc[-1] * math.sqrt(
        HORIZON_SESSIONS["1W"]
    )

    assert horizon_sigma(closes[::-1], "1W") == pytest.approx(expected)


def test_recent_volatility_dominates_the_band():
    """Calm for a year, then three volatile weeks: the band must reflect the
    recent weeks. Reading the series in the wrong direction would make the
    volatile weeks the oldest observations and shrink the band."""
    calm = [100.0 * (1.001 if i % 2 else 1.0) for i in range(230)]
    volatile = [100.0 * (1.05 if i % 2 else 1.0) for i in range(20)]
    oldest_first = calm + volatile

    newest_first_sigma = horizon_sigma(oldest_first[::-1], "1W")
    wrong_order_sigma = horizon_sigma(oldest_first, "1W")

    assert newest_first_sigma > 3 * wrong_order_sigma
