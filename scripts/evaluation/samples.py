"""Evaluation sample grid: which (ticker, as_of_date) pairs get a debate."""

from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

TICKERS = ["AAPL", "CAT", "GS", "INTC", "META", "NFLX", "NVDA", "SMCI", "TSLA"]


@dataclass(frozen=True)
class Sample:
    ticker: str
    as_of: date
    horizon: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Join key shared by labels and predictions (all strings, JSON-safe)."""
        return (self.ticker, self.as_of.isoformat(), self.horizon)


def build_grid(
    start: date, months: int, horizon: str, tickers: list[str] = TICKERS
) -> list[Sample]:
    """One sample per ticker per month: as_of = start, start+1M, ..."""
    return [
        Sample(ticker, start + relativedelta(months=i), horizon)
        for ticker in tickers
        for i in range(months)
    ]
