"""Frontend settings — all overridable through environment variables.

`API_URL` is set by docker-compose (`http://backend:8000`). When it is not
set (plain `streamlit run frontend/app.py` on a dev machine) the app talks to
a backend on `http://localhost:8000`.
"""

import os

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"
# Fallback only — times are shown in the browser's zone when it reports one.
TIMEZONE = os.getenv("TZ", "")
REQUEST_TIMEOUT_S = 10
HEALTH_TIMEOUT_S = 3

# Display labels for the horizon codes from GET /meta.
HORIZONS = {"1W": "1 week", "1M": "1 month", "3M": "3 months"}

POLL_SECONDS = 2
