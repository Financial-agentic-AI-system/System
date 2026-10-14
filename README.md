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
| `-f` | `--functions` | **Yes** | Comma-separated list of Alpha Vantage API functions (e.g., `INCOME_STATEMENT,TIME_SERIES_DAILY`). | *None* |
| `-t` | `--tickers` | No | Comma-separated list of stock symbols (e.g., `AAPL,TSLA`). | `AAPL,MSFT,NVDA,JPM,GS...` |
| `-p` | `--period` | No | Filter for fundamental data reports. Available options: `annual`, `quarterly`, `all`. | `all` |

#### Data Ingestion
The system includes a generic CLI tool for fetching data from Alpha Vantage.

Basic usage:

```bash
python data_scripts/alpha_vantage_fetcher.py --functions <FUNCTIONS> --tickers <TICKERS> --period <PERIOD>
```
#### Loading data into db

In order to load data into db use: 
```bash
uv run python -m src.db.load
```
This command will create tables and load data directly from Datalake.