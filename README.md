# Multi-Agent Financial System (MAS)

An AI-driven investment analysis system using a Multi-Agent architecture to simulate hedge fund decision-making. This project leverages Large Language Models (LLMs) to process macroeconomic, fundamental, and sentiment data.

## Project Structure

For the system to work correctly, maintain the following directory structure:
```text
Thesis/
├── code/              <-- This repository (logic)
└── datalake/          <-- External data repository (JSON storage)
```
## Setup

1. **Install uv** (if you haven't already):
   Follow the instructions at [astral.sh/uv](https://astral.sh/uv/).

2. **Set up the environment and install dependencies**:
   ```bash
   git clone <repository_url>
   cd <repository_name>

   # Create a virtual environment
   uv venv

   # Sync dependencies from pyproject.toml
   uv sync
   ```

## Development & Running

### Using Docker (Recommended)
To run the full stack (Redis, Postgres/pgvector, backend, worker, frontend), use:

```bash  
docker compose up --build
```

### Services and ports

| Service | Address | What it is |
| --- | --- | --- |
| Frontend | http://localhost:8501 | Streamlit app: start a debate, watch it live, see the result |
| Backend API | http://localhost:8000 | FastAPI; interactive docs at http://localhost:8000/docs |
| Postgres (pgvector) | `localhost:5432` | user `user`, password `password`, database `mas_db` |
| Redis | `localhost:6379` | Celery broker and live debate cache |
| Worker | — | Celery worker that runs the debates; no port |

### Using the system

1. Start the stack: `docker compose up --build`.
2. Load the data into the database (first run, or after the datalake changes):
   `uv run python -m src.db.load`.
3. Open http://localhost:8501, pick a ticker, horizon and as-of date, and
   click **Start debate**.

The same can be done through the API:

```bash
curl -X POST http://localhost:8000/api/v1/predict/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon": "1W", "as_of_date": "2026-01-15"}'
curl http://localhost:8000/api/v1/predict/result/<task_id>
```

To work on the frontend without rebuilding its container, run everything else
in Docker and the app locally:

```bash
docker compose up --build postgres redis backend worker
uv run streamlit run frontend/app.py
```

### Debate logs

The worker logs one line per LLM call (node, tokens, cost, time) and a summary
per debate (decision, rounds, totals):

```bash
docker compose logs -f worker
```

`LOG_LEVEL=DEBUG` in `.env` also logs the full prompts and responses. Prices
for the cost are set in `.env` (`LLM_PRICE_INPUT_PER_M`, `LLM_PRICE_OUTPUT_PER_M`).

## Local Development

Always use `uv` for running tasks to ensure consistency with the production environment:

**Tests**:
```bash
uv run pytest tests/unit
```

#### Available CLI Flags

The ingestion tool supports the following arguments:

| Short | Long Flag | Required | Description | Default |
| :--- | :--- | :---: | :--- | :--- |
| `-f` | `--functions` | **Yes** | Comma-separated list of financial statements: `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`. | *None* |
| `-t` | `--tickers` | No | Comma-separated list of stock symbols (e.g., `AAPL,TSLA`). | `AAPL,MSFT,NVDA,JPM,GS...` |
| `-p` | `--period` | No | Filter for fundamental data reports. Available options: `annual`, `quarterly`, `all`. | `all` |

#### Data Ingestion
The system includes a generic CLI tool for fetching data from Alpha Vantage.

Basic usage:

```bash
python scripts/data/alpha_vantage_fetcher.py --functions <FUNCTIONS> --tickers <TICKERS> --period <PERIOD>
```
#### Loading data into db

In order to load data into db use: 
```bash
uv run python -m src.db.load
```
This command will create tables and load data directly from Datalake.
After the articles are loaded, the ones without an embedding are vectorized
(needs `GCP_PROJECT_ID` and `gcloud auth application-default login`; skipped
when `GCP_PROJECT_ID` is not set).

```bash
uv run python -m src.db.load --datasets fundamentals prices   # only some datasets
uv run python -m src.db.load --datalake ../Datalake --no-init # custom path, skip table creation
uv run python -m src.db.load --datasets articles              # (re)load articles + embed new ones
uv run python -m src.db.load --no-embed                       # skip the embedding step
```

## Backtest evaluation

Measures how often the debate's BUY/HOLD/SELL matches the Triple Barrier label
of the real price path. Method, sample grid and limitations:
[docs/evaluation.md](docs/evaluation.md). Labels, predictions and metrics are
written to `datalake/evaluation`.

Prerequisites: the datalake loaded into Postgres (see above, including the
article embeddings), `.env` with `GCP_PROJECT_ID` and `GCP_LOCATION` (the region
must match `docker/backtest-proxy/squid.conf`) and
`gcloud auth application-default login`. Run everything from this directory.

```bash
# 0. services
docker compose up -d postgres redis

# 1. labels: 10 tickers x 19 monthly dates, horizon 1M (no LLM, one file per k)
uv run python -m scripts.evaluation.evaluate labels --horizon 1M --k 1

# 2. rebuild the worker image (src/ is baked into it; needed after any change in src/)
docker compose --profile backtest build backtest-worker

# 3. predictions: the debates, behind the allowlisting proxy.
#    Try 5 samples first, then run it again without --limit. The same --run-id resumes.
docker compose --profile backtest run --rm backtest-worker uv run --no-sync python -m scripts.evaluation.evaluate predict --run-id r1 --limit 5
docker compose --profile backtest run --rm backtest-worker uv run --no-sync python -m scripts.evaluation.evaluate predict --run-id r1

# 4. metrics: reads the predictions and the labels files, prints and saves metrics_r1.json
uv run python -m scripts.evaluation.evaluate metrics --predictions ../datalake/evaluation/predictions_r1.jsonl --k 1
```

