"""Volatility band that defines the BUY/HOLD/SELL labels for a horizon.

The same band is used by the Portfolio Manager prompt and by the backtest
labelling (scripts/evaluation), so the model is judged by the rule it was told.

    sigma_h = ewm_std(daily returns, span=100) * sqrt(sessions in the horizon)

The daily volatility is the exponentially weighted std of López de Prado's
`getDailyVol` (Advances in Financial Machine Learning, ch. 3), i.e.
`returns.ewm(span=100).std()` in pandas. The book uses that daily figure
directly as the barrier width; we additionally scale it by sqrt(n) so the
band matches the length of the horizon.

Only closes up to `as_of_date` may be passed in (point-in-time).
"""

import math
from collections.abc import Sequence

# Trading sessions per horizon, and how the horizon is spelled out in prompts.
HORIZON_SESSIONS: dict[str, int] = {"1W": 5, "1M": 21, "3M": 63}
HORIZON_LONG: dict[str, str] = {
    "1W": "1 week (about 5 trading sessions)",
    "1M": "1 month (about 21 trading sessions)",
    "3M": "3 months (about 63 trading sessions)",
}
VOL_SPAN = 100  # EWM span, as in getDailyVol
# EWM weights are never zero, so truncate far enough that the tail is negligible:
# (1 - 2/(span+1)) ** 250 is < 1% of the total weight for span 100.
HISTORY_SESSIONS = 250


def ewm_std(values_desc: Sequence[float], span: int) -> float:
    """Exponentially weighted std, equal to pandas `Series.ewm(span=span).std()`
    evaluated at the newest observation (adjust=True, bias-corrected).

    `values_desc` is newest-first; observation i gets weight (1 - alpha) ** i.
    """
    alpha = 2 / (span + 1)
    w = [(1 - alpha) ** i for i in range(len(values_desc))]
    sum_w = sum(w)
    sum_w2 = sum(x * x for x in w)
    mean = sum(wi * v for wi, v in zip(w, values_desc)) / sum_w
    var = sum(wi * (v - mean) ** 2 for wi, v in zip(w, values_desc)) / sum_w
    return math.sqrt(var / (1 - sum_w2 / sum_w**2))


def horizon_sigma(
    closes_desc: Sequence[float], horizon: str, span: int = VOL_SPAN
) -> float | None:
    """1-sigma move over `horizon` as a fraction (0.08 = 8%).

    `closes_desc` is newest-first (the order `fetch_prices` returns). Returns
    None when there are too few prices to estimate it.
    """
    closes = list(closes_desc)
    if len(closes) < 3:
        return None
    returns = [closes[i] / closes[i + 1] - 1 for i in range(len(closes) - 1)]
    return ewm_std(returns, span) * math.sqrt(HORIZON_SESSIONS[horizon])
