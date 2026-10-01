"""Frontend settings — all overridable through environment variables.

`API_URL` is set by docker-compose (`http://backend:8000`). When it is not
set (plain `streamlit run frontend/app.py` on a dev machine) the app talks to
a backend on `http://localhost:8000`.
"""

import os
from datetime import date

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"
# Fallback only — times are shown in the browser's zone when it reports one.
TIMEZONE = os.getenv("TZ", "")
REQUEST_TIMEOUT_S = 10
HEALTH_TIMEOUT_S = 3

# The datalake only holds 9 tickers and there is no "list tickers"
# endpoint, so the selector is configured here (comma-separated env var).
DEFAULT_TICKERS = "AAPL,CAT,GS,INTC,META,NFLX,NVDA,SMCI,TSLA"
TICKERS = [
    t.strip().upper()
    for t in os.getenv("TICKERS", DEFAULT_TICKERS).split(",")
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
