"""Frontend settings — all overridable through environment variables.

`API_URL` is set by docker-compose (`http://backend:8000`). When it is not
set (plain `streamlit run frontend/app.py` on a dev machine, backend not
running yet) the app starts in mock mode; see `default_mock()`.
"""

import os
from datetime import date

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"
REQUEST_TIMEOUT_S = 10

# The datalake only holds ~9-10 tickers and there is no "list tickers"
# endpoint, so the selector is configured here (comma-separated env var).
TICKERS = [
    t.strip().upper()
    for t in os.getenv("TICKERS", "AAPL,MSFT,NVDA,JPM,GS,TSLA").split(",")
    if t.strip()
]

# horizon code sent to the backend -> label shown to the user
HORIZONS = {"1W": "1 week", "1M": "1 month", "3M": "3 months"}

# Backtest window from docs/evaluation.md §3.3 — the model's training cutoff
# (Dec 2023) only guarantees a clean, leak-free run inside this window.
BACKTEST_START = date(2025, 1, 1)
BACKTEST_END = date(2026, 6, 30)

MAX_ROUNDS = 3  # mirrors src/agents/graph.py::MAX_ROUNDS
POLL_SECONDS = 2


def default_mock() -> bool:
    """MOCK_MODE wins if set; otherwise mock only when API_URL is not set."""
    flag = os.getenv("MOCK_MODE")
    if flag is not None:
        return flag.lower() in {"1", "true", "yes"}
    return "API_URL" not in os.environ
