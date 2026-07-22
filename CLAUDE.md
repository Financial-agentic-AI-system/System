# Multi-Agent Financial System (MAS)

AI-driven investment analysis system that simulates short and long term decision-making
with a multi-agent LLM architecture (engineering thesis project). Agents debate
a stock's outlook — financial fundamentals, sentiment, macro context — under a
Portfolio Manager and a Critic, then the system returns a prediction.

This repo (`code/`) is one of two sibling repos under `Thesis/`:

- `Thesis/code/` — this repo (logic)
- `Thesis/datalake/` — external repo, raw JSON data, mounted read-only at
  `/datalake` inside the `backend` container

See @README.md for setup instructions, @docs/ARCHITECTURE.md for the full
system diagram and REST/prediction contract, @docs/agents.md for the exact
debate protocol (round limit, termination condition), and
@docs/evaluation.md for the backtesting methodology.

## Tech stack

- Python 3.12, dependency manager: **uv** — never use bare `pip install` or
  `poetry`, always `uv sync` / `uv run`.
- FastAPI (backend), LangGraph (agent debate graph), Celery + Redis (async
  task queue), SQLAlchemy + Postgres/pgvector (vector store),
  LangChain (retriever), Streamlit (frontend).
- LLM: Gemini Enterprise Agent Platform, via `src/agents/llm_client.py`.
- Embeddings: Vertex AI `textembedding-gecko@003`, via
  `src/retriever/embeddings.py`.
- Linting/formatting: **ruff only**.

## Repository layout

| Path                 | Purpose                                                                |
| -------------------- | ----------------------------------------------------------------------- |
| `data_scripts/`      | one-off CLI scripts; fetch raw data from Alpha Vantage into the datalake |
| `notebooks/`         | experiments, not production code                                        |
| `src/agents/`        | LangGraph debate: `states.py`, `nodes.py`, `graph.py`, prompts in `prompts/*.yaml` via `prompt_loader.py` |
| `src/api/v1/`        | FastAPI routes — 3 endpoints, see `endpoints/`                          |
| `src/worker/`        | the ONLY Celery worker in this system — runs the LangGraph debate       |
| `src/transformer/`   | one-off data transformer — NOT a Celery worker (see rule below)         |
| `src/retriever/`     | LangChain semantic search + pgvector upsert                             |
| `src/db/`            | SQLAlchemy models, session, Alembic migrations                          |
| `src/cache/`         | Redis client for live debate/agent status                               |
| `frontend/`          | Streamlit app, talks to the backend over REST                          |
| `docker/`            | one Dockerfile per service (backend, worker, frontend)                  |
| `docs/adr/`          | short decision records — check before overturning an existing architectural choice |

## Critical rules

- **IMPORTANT**: `src/transformer/` is a plain one-off script, never a Celery
  task and never triggered by the backend. It's run manually by a developer:
  `python -m src.transformer.transform --tickers AAPL,TSLA`. Same for data
  ingestion (`data_scripts/*`) — there is no API endpoint for either; this is
  a deliberate scope cut, not a gap (see `docs/ARCHITECTURE.md`).
- **IMPORTANT**: agent prompts live in `src/agents/prompts/*.yaml` and are
  read through `prompt_loader.py`. Never hardcode a system prompt as a Python
  string inside `nodes.py`.
- `src/worker/celery_app.py` is the single Celery app for the whole project.
- Never commit `.env`; only `.env.example` is tracked. Real secrets (Alpha
  Vantage key, Vertex AI credentials, Gemini key) stay local or in CI
  secrets.
- The datalake is read-only from this repo's point of view. Never write into
  `/datalake`; derived data (embeddings, metadata) goes into Postgres/pgvector
  through `src/retriever/vectorstore.py`.

## Commands

Local environment setup:

```bash
uv venv
uv sync
```

Run everything with Docker (Redis, Postgres/pgvector, backend, worker, frontend):

```bash
docker compose up --build
```

Run the data transformer manually (reuses the `backend` image, no dedicated service):

```bash
docker compose run --rm backend uv run python -m src.transformer.transform --tickers AAPL,TSLA
```

Fetch raw data from Alpha Vantage into the datalake:

```bash
uv run python data_scripts/alpha_vantage_fetcher.py --functions INCOME_STATEMENT,TIME_SERIES_DAILY --tickers AAPL,TSLA --period annual
```

Run unit tests:

```bash
uv run pytest tests/unit
```

Lint and format (run both before committing):

```bash
uv run ruff check .
uv run ruff format .
```

## Conventions

- Type hints everywhere in `src/`. Use `list[str] | None`, not
  `Optional[List[str]]`.
- One Pydantic schema pair per endpoint group in `src/api/v1/schemas.py`
  (`XRequest` / `XResponse`).
- When unsure between two implementation approaches for agent/graph/worker
  code, explain both and ask — don't silently pick one.
- If function is critical write unit tests.
