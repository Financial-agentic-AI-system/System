# Testing

The repository has three kinds of tests:

- **Unit tests** — `tests/unit/`, run by pytest locally and in CI. They need
  no running service: no Postgres, no Redis, no Celery broker, no LLM and no
  embedding API.
- **Integration tests** — the pytest files in `tests/integration/`. They run
  against a real Postgres with pgvector and a real Redis and check real
  objects end to end; the LLM is the only stand-in.
- **Smoke scripts** — the remaining scripts in `tests/integration/`, run by
  hand against a real database and a real LLM.

Plain `uv run pytest` collects only the unit tests
(`testpaths = ["tests/unit"]` in `pyproject.toml`).

## 1. Running

```bash
uv run pytest tests/unit
```

Coverage (line and branch) can be measured without adding a dependency to the
project:

```bash
uv run --with pytest-cov pytest tests/unit \
    --cov=src --cov=scripts --cov=frontend --cov-branch --cov-report=term-missing
```

Integration tests need Postgres with pgvector and Redis, e.g. the services from
`docker-compose.yml`:

```bash
docker compose up -d postgres redis
uv run pytest tests/integration
```

| Variable | Default |
| --- | --- |
| `TEST_DATABASE_URL` | `postgresql://user:password@localhost:5432/mas_test` |
| `TEST_REDIS_URL` | `redis://localhost:6379/15` |

The database name must end in `_test` (every test empties all tables, so the
application database `mas_db` is never touched); it is created on first use.
Redis uses its own logical database and flushes only that one. When a service
cannot be reached the tests are skipped, or fail when the `CI` environment
variable is set.

CI (`.github/workflows/unit-tests.yaml`) runs the unit tests on every push and
pull request to `main`. A second job checks that the backend, the worker and
the frontend import with only the dependency groups their Docker images
install.

## 2. How external services are replaced

| Service | Replacement in unit tests |
| --- | --- |
| Redis | `fakeredis`, patched into `src.cache.debate_cache.redis_client` |
| Postgres (reads) | `get_session` and the `fetch_*` queries patched to return no rows or canned rows |
| Postgres (SQL shape) | statements compiled with the PostgreSQL dialect and checked as text (`WHERE`, `ORDER BY`, `ON CONFLICT … DO UPDATE`) |
| Celery broker | `celery_app.send_task` replaced by a recorder; the task itself runs eagerly through `.apply()` |
| LLM | `llm_client.generate_structured` / `generate_tool_calls` replaced by scripted fakes; HTTP-level tests patch `requests.post` |
| Embedding API, Google auth | `requests.post` and `google.auth.default` patched with fake responses and credentials |
| Backend (for the frontend) | `requests.request` / `requests.get` patched with queued responses |

Shared fixtures live in `tests/conftest.py`:

- `fake_redis` — a fresh fakeredis instance per test.
- `no_real_db` — every query the debate nodes make returns no rows, and the
  Sentiment agent's search step answers "no search needed".
- `debate_state` — factory for a round-1 `DebateState`; keyword arguments
  override keys.
- `fake_llm` — factory that installs a scripted `generate_structured`: the
  Critic disagrees for its first N calls, the PM re-asks a chosen set of
  agents, and one chosen call can be made to raise. It counts calls per output
  schema.

The same conftest adds `frontend/` to `sys.path`, so frontend modules are
imported by bare name, as they are inside the frontend container.

## 3. Unit tests by module

### Agents (`src/agents/`)

| File | What it checks |
| --- | --- |
| `test_agents_graph.py` | The compiled debate graph end to end with a scripted LLM: termination after round 1 when the Critic agrees, a partial re-ask of only the agents the PM selects, the stop at the round limit when the Critic keeps disagreeing; `route_to_agents` creates one `Send` per active agent. |
| `test_agents_nodes.py` | Context formatters (fundamentals, macro trend, reports, PM opinion); which prompt key each node picks in round 1 and in later rounds, and what it passes into it; the PM's HOLD band from price volatility; `finalize_node` assembling `PredictResult`; `horizon_sigma` scaling with √sessions and `ewm_std` matching pandas `ewm(span).std()`. |
| `test_prompt_loader.py` | Placeholder substitution, the default `system_prompt` key, clear errors for a missing key, a missing placeholder or an unknown prompt file, YAML caching. |
| `test_llm_client.py` | Parsing of structured answers, including Llama 4 answers wrapped as a function call. |
| `test_usage.py` | Token and cost accounting per debate (also across LangGraph's worker threads), log lines with the debate id, recording of successful, failed and retried LLM calls. |

### API, cache and worker

| File | What it checks |
| --- | --- |
| `test_api_v1.py` | All REST endpoints: request validation (horizon, ticker, date, future date), unknown tickers, 503 when the database or the broker is down (the task is then marked `FAILED`), 404 for unknown tasks, status/history/result envelopes for every task state, `/meta` window from the price range, `/tickers`, `/health`. The cache is seeded through `debate_cache`, so the API is tested against exactly what the worker writes. |
| `test_debate_cache.py` | Status, history, result and metadata roundtrips in Redis, TTLs, history keeping both reports when an agent is re-asked. |
| `test_worker_tasks.py` | `execute_debate` mirroring the graph into the cache: `DONE` status, result and ordered history with round numbers; `FAILED` status with the partial history kept when an LLM call raises; the Celery wrapper, including the ISO-string date the API sends; the summary log line. |

### Database and retrieval (`src/db/`, `src/retriever/`)

| File | What it checks |
| --- | --- |
| `test_db_loaders.py` | Cleaning of raw Alpha Vantage values, date parsing, the datalake iterators for fundamentals, prices, articles and FRED series (skipping bad rows), datalake path resolution. |
| `test_article_embeddings.py` | Embedding only the missing vector columns, the text each variant is built from, commits per chunk, and the upsert keeping existing vectors unless the article text changed. |
| `test_article_search.py` | The pgvector similarity query: ticker filter, point-in-time bound (`time_published < as_of + 1 day`), cosine distance ordering, similarity threshold, defaults; recent-articles query not loading vectors; the retriever embedding the search text as a query. |
| `test_embeddings.py` | The Vertex AI request (endpoint, task type, dimensionality), batching, one text per request for Gemini embeddings, retry on transient errors and none on permanent ones, dimension check, token reuse and refresh retries, a hint when ADC credentials are rejected. |

### Evaluation (`scripts/evaluation/`)

| File | What it checks |
| --- | --- |
| `test_evaluation_labels.py` | Triple Barrier: the first barrier touched decides, a touch exactly on a barrier and on the last session counts, HOLD at the vertical barrier, the width scales with `k`; `horizon_sigma` on newest-first closes equals pandas on calendar-ordered closes, and recent volatility dominates the band. |
| `test_evaluation_cli.py` | `evaluate metrics`: joining predictions with the labels file, counting failed and unlabelled predictions, the saved metrics JSON, a clear error naming the missing labels file; compaction of the predictions file. |

### Frontend (`frontend/`)

| File | What it checks |
| --- | --- |
| `test_frontend_api_client.py` | The REST client: request bodies, response parsing, readable errors for 404/422/503/500, an unreachable backend and unexpected response shapes. |
| `test_frontend_debate_model.py` | Grouping the history into rounds, the per-stage state shown in the live view, the progress message for each phase of a debate. |
| `test_frontend_models_match_backend.py` | The frontend's copy of the response models has the same fields, directions and statuses as the backend's. |

## 4. Integration tests

Fixtures in `tests/integration/conftest.py`: `db_session` (the empty test
database; production `get_session()` points at it too), `redis_db` (the debate
cache on the test Redis) and `load_prices` (writes closes as a datalake file
and loads them with `src/db/loaders.py`).

| File | What it checks |
| --- | --- |
| `test_frontend_backend.py` | The frontend's `BackendClient` against the FastAPI app served by uvicorn: tickers and the backtest window from the database, a debate started through the API and run by a Celery worker over Redis (the Critic disagrees once, so round 2 re-asks only Macro), status, history and result read back and turned into rounds by `debate_model`. Every response is also fetched raw and must survive the frontend models unchanged. Backend errors (unknown ticker, future date, unknown task) reach the frontend as `ApiError`. The LLM answers from a script. |
| `test_labels_from_database.py` | `build_labels` on prices loaded into Postgres: p0 is the close on `as_of` (or the last one before a weekend `as_of`), the path starts on the next session, the first barrier touched decides, a move after the vertical barrier is ignored, wider `k` turns a touch into HOLD, samples without enough prices are skipped with a reason. `sigma_h` is checked against pandas on the closes in calendar order, and the Portfolio Manager's prompt quotes the same band from the agents' own price query. |

## 5. Smoke scripts

The scripts below call the real services and cost LLM tokens.
Each needs credentials in `.env` (see `.env.example`); the two that read the
database also need Postgres running with the datalake loaded
(`uv run python -m src.db.load`). Usage is in each script's docstring.

| Script | What it runs |
| --- | --- |
| `test_llm_client.py` | A few real calls to the LLM endpoint: authentication, structured output, same answer twice at temperature 0. |
| `test_agent_db_integration.py` | One agent node (Financial) on real database rows and a real LLM call. |
| `test_full_debate.py` | The full compiled debate graph on real data and a real LLM. |
