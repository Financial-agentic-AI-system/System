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

## Development & Running

### Using Docker (Recommended)
To run the full stack (Redis, Postgres/pgvector, backend, worker, frontend), use:

```bash
docker compose up --build
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