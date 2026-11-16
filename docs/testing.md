# Testing

The repository has two kinds of tests:

- **Unit tests** — `tests/unit/`, run by pytest locally and in CI. They need
  no running service: no Postgres, no Redis, no Celery broker, no LLM and no
  embedding API.
- **Integration tests** — `tests/integration/`. They run against a real
  Postgres with pgvector and a real Redis and check real objects end to end.
  Most of them script the LLM's answers; the ones marked `llm` call the real
  LLM and embedding API.

Plain `uv run pytest` collects only the unit tests
(`testpaths = ["tests/unit"]` in `pyproject.toml`), and tests marked `llm` are
deselected unless asked for (`addopts = "-v -m 'not llm'"`).

## 1. Running

```bash
uv run pytest tests/unit
```

Coverage (line and branch, of `src/`, `scripts/` and `frontend/`; settings in
`[tool.coverage.run]` of `pyproject.toml`):

```bash
uv run pytest tests/unit --cov --cov-report=term-missing
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

Tests marked `llm` cost tokens and need the credentials from `.env` (see
`.env.example`) plus Application Default Credentials
(`gcloud auth application-default login`). They are skipped without
`GCP_PROJECT_ID` or `GEMINI_API_KEY`:

```bash
uv run --env-file .env pytest tests/integration -m llm
```

The database name must end in `_test` (every test empties all tables, so the
application database `mas_db` is never touched); it is created on first use.
Redis uses its own logical database and flushes only that one. When a service
cannot be reached the tests are skipped, or fail when the `CI` environment
variable is set.

CI (`.github/workflows/unit-tests.yaml`) runs on every push and pull request to
`main`, in three jobs:

- `test` — the unit tests.
- `integration-tests` — unit and integration tests together, against
  `pgvector/pgvector:pg16` and `redis:7-alpine` service containers, with
  coverage of both; the report is printed in the log and in the run's summary.
  Tests marked `llm` are not run.
- `image-dependencies` — the backend, the worker and the frontend import with
  only the dependency groups their Docker images install.

## 2. How external services are replaced

| Service | Replacement in unit tests |
| --- | --- |
| Redis | `fakeredis`, patched into `src.cache.debate_cache.redis_client` |
| Postgres (reads) | `get_session` and the `fetch_*` queries patched to return no rows or canned rows |
| Postgres (SQL shape) | statements compiled with the PostgreSQL dialect and checked as text (`WHERE`, `ORDER BY`, `ON CONFLICT … DO UPDATE`) |
| Celery broker | `celery_app.send_task` replaced by a recorder; the task itself runs eagerly through `.apply()` |
| LLM | `llm_client.generate_structured` / `generate_tool_calls` replaced by scripted fakes; the client itself is tested against a scripted HTTP server on localhost |
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
| `test_llm_client.py` | Parsing of structured answers, including Llama 4 answers wrapped as a function call. Against a scripted HTTP server on localhost: the request body (schema, temperature 0, no tools on structured calls), retries of server errors, dropped connections, non-JSON bodies and answers that miss the schema with `2 ** attempt` waits, `Retry-After` and the longer backoff on 429, no retry on 400/401/403/404, giving up after 5 attempts, tool calls parsed and malformed ones skipped. |
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
| `test_debate_with_llm.py` (`llm`) | A fictional company written as datalake files and loaded with the real loaders (fundamentals, prices, FRED series, articles, one of them published after `as_of_date`); the articles are embedded with the real embedding API. The Financial agent answers from the database with one real LLM call; a full debate on the real LLM ends `DONE` with valid values, the protocol order in the Redis history, and no article from after `as_of_date` among the Sentiment agent's search results. Opinions are never checked, since the model is not deterministic. |
| `test_point_in_time.py` | The point-in-time bounds of `src/db/queries.py` on a loaded datalake, day by day around each boundary: a report is visible from its `reported_date`, or from 90 days after its fiscal date when the date is missing, and `quarters` keeps the newest published quarters with all three statements; prices and macro values up to and including `as_of_date`; articles published until the end of `as_of_date` and never later, for recent articles, articles re-read by id, the count of missing vectors and the pgvector similarity search (also never another ticker's). |
| `test_loading.py` | Loading the datalake twice with the real loaders: no duplicate rows and changed values updated; article vectors kept on reload, the title+summary vector reset when the title changed, both when the summary changed, and the next embedding run sends only the reset ones. |
| `test_labels_from_database.py` | `build_labels` on prices loaded into Postgres: p0 is the close on `as_of` (or the last one before a weekend `as_of`), the path starts on the next session, the first barrier touched decides, a move after the vertical barrier is ignored, wider `k` turns a touch into HOLD, samples without enough prices are skipped with a reason. `sigma_h` is checked against pandas on the closes in calendar order, and the Portfolio Manager's prompt quotes the same band from the agents' own price query. |
