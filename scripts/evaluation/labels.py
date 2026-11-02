"""Ground-truth labels via the Triple Barrier Method (López de Prado, AFML ch. 3).

Horizontal barriers sit at p0 * (1 +/- k * sigma_h), where sigma_h is the same
`horizon_sigma` the Portfolio Manager prompt quotes, so the model is judged by
the rule it was told. The vertical barrier is the end of the horizon (n sessions
after as_of): if neither horizontal barrier is touched, the label is HOLD.

Labels use information from AFTER as_of on purpose. They must never be written
to the tables the agents read, nor passed into a debate.
"""

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from scripts.evaluation.samples import Sample
from src.agents.volatility import (
    HISTORY_SESSIONS,
    HORIZON_SESSIONS,
    VOL_SPAN,
    horizon_sigma,
)
from src.db.models import StockPrice


def triple_barrier(
    p0: float, path: Iterable[tuple[date, float]], sigma_h: float, k: float
) -> tuple[str, date | None]:
    """Label of the first horizontal barrier touched on daily closes.

    `path` is the closes after as_of, oldest first, up to the vertical barrier.
    Returns (label, touch_date); touch_date is None for HOLD.
    """
    upper = p0 * (1 + k * sigma_h)
    lower = p0 * (1 - k * sigma_h)
    for day, close in path:
        if close >= upper:
            return "BUY", day
        if close <= lower:
            return "SELL", day
    return "HOLD", None


def load_closes(session: Session, ticker: str) -> list[tuple[date, float]]:
    """Whole closing-price series of a ticker, oldest first."""
    rows = session.execute(
        select(StockPrice.date, StockPrice.close)
        .where(StockPrice.symbol == ticker)
        .order_by(StockPrice.date)
    ).all()
    return [(d, float(c)) for d, c in rows]


def _key(s: Sample) -> dict:
    return {"ticker": s.ticker, "as_of": s.as_of.isoformat(), "horizon": s.horizon}


def build_labels(
    session: Session, samples: list[Sample], k: float, span: int = VOL_SPAN
) -> tuple[list[dict], list[dict]]:
    """Returns (labels, skipped). `skipped` says why a sample got no label."""
    by_ticker: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_ticker[s.ticker].append(s)

    labels: list[dict] = []
    skipped: list[dict] = []
    for ticker, ticker_samples in by_ticker.items():
        series = load_closes(session, ticker)
        dates = [d for d, _ in series]
        closes = [c for _, c in series]

        for s in ticker_samples:
            n = HORIZON_SESSIONS[s.horizon]
            idx = bisect_right(dates, s.as_of)  # series[:idx] is known at as_of
            if idx == 0:
                skipped.append({**_key(s), "reason": "no price on/before as_of"})
                continue
            path = series[idx : idx + n]
            if len(path) < n:
                skipped.append({**_key(s), "reason": "horizon outruns price data"})
                continue

            history_desc = closes[max(0, idx - HISTORY_SESSIONS) : idx][::-1]
            sigma_h = horizon_sigma(history_desc, s.horizon, span)
            if sigma_h is None:
                skipped.append({**_key(s), "reason": "too little price history"})
                continue

            p0 = closes[idx - 1]
            label, touch = triple_barrier(p0, path, sigma_h, k)
            labels.append(
                {
                    **_key(s),
                    "p0": p0,
                    "sigma_h": sigma_h,
                    "upper": p0 * (1 + k * sigma_h),
                    "lower": p0 * (1 - k * sigma_h),
                    "label": label,
                    "touch_date": touch.isoformat() if touch else None,
                    "ret_end": path[-1][1] / p0 - 1,  # final return, for reference
                    "k": k,
                    "span": span,
                }
            )
    return labels, skipped
