# Database

Postgres 16 with the pgvector extension (`pgvector/pgvector:pg16` in
`docker-compose.yml`). It holds everything the agents read during a debate:
financial statements, daily prices, FRED macro series and news articles with
their embeddings. Debate state and transcripts are not stored here; they live
in Redis (`src/cache/`).

Code: `src/db/`. A short module overview and instructions for connecting from
VS Code or `psql` are in `src/db/README.md`.

## 1. Data flow

```mermaid
flowchart LR
    av[[Alpha Vantage / yfinance / FRED]] -->|scripts/data/*, manual| lake[(datalake/raw_data<br/>read-only)]
    lake -->|python -m src.db.load| loaders[src/db/loaders.py<br/>parse + upsert]
    loaders --> pg[(Postgres / pgvector)]
    loaders -->|articles only| emb[src/db/article_embeddings.py]
    emb -->|text-embedding-005| vertex[[Vertex AI]]
    emb -->|vectors| pg
    pg -->|src/db/queries.py<br/>point-in-time reads| agents[Agents, API, retriever]
```

- **Write side** — `src/db/load.py` (CLI), `src/db/loaders.py`,
  `src/db/article_embeddings.py`. Run by hand, never by the backend or Celery.
- **Read side** — `src/db/queries.py`. All SQL used by the agents, the API and
  the retriever lives there; nothing outside `src/db/` writes queries.

## 2. Tables

![Database tables](mas_db_diagram.png)

The diagram predates `fundamentals.reported_date`; the tables below are
authoritative (`src/db/models.py`).

All tables have a surrogate `id` primary key and a natural unique key, which
the loaders use for upserts.

### `fundamentals`

One row per financial report: one ticker, one report type, one period, one
fiscal date. Hybrid shape: key columns plus the whole report in JSONB.

| Column          | Type          | Notes                                                        |
| --------------- | ------------- | ------------------------------------------------------------ |
| `id`            | `integer`     | PK                                                           |
| `symbol`        | `varchar(16)` | indexed                                                      |
| `report_type`   | `varchar(32)` | `BALANCE_SHEET`, `CASH_FLOW`, `INCOME_STATEMENT`; indexed    |
| `period`        | `varchar(16)` | `annual`, `quarterly` (`unknown` if the file has none)       |
| `fiscal_date`   | `date`        | `fiscalDateEnding`, end of the reporting period; indexed     |
| `reported_date` | `date`, null  | publication date (Alpha Vantage `EARNINGS.reportedDate`); indexed. `NULL` for files from the previous datalake version |
| `data`          | `jsonb`       | the full report, values normalised (see §3.2)                |

Unique key: `(symbol, report_type, period, fiscal_date)` — `uq_fundamentals_key`.

### `stock_prices`

| Column   | Type             | Notes          |
| -------- | ---------------- | -------------- |
| `id`     | `integer`        | PK             |
| `symbol` | `varchar(16)`    | indexed        |
| `date`   | `date`           | indexed        |
| `close`  | `numeric(20, 6)` | closing price  |

Unique key: `(symbol, date)` — `uq_stock_price_key`. This table also defines
which tickers exist (`GET /tickers`) and the as-of date range offered by
`GET /meta`.

### `article_summaries`

| Column                    | Type           | Notes                                       |
| ------------------------- | -------------- | ------------------------------------------- |
| `id`                      | `integer`      | PK                                          |
| `symbol`                  | `varchar(16)`  | ticker the article was fetched for; indexed |
| `title`                   | `varchar`      |                                             |
| `url`                     | `varchar`      |                                             |
| `time_published`          | `timestamp`    | Alpha Vantage `time_published`              |
| `source`                  | `varchar`      |                                             |
| `summary`                 | `varchar`      |                                             |
| `overall_sentiment_score` | `float`        |                                             |
| `overall_sentiment_label` | `varchar(32)`  |                                             |
| `ticker_sentiment_score`  | `float`        | sentiment towards `symbol`                  |
| `ticker_sentiment_label`  | `varchar(32)`  |                                             |
| `relevance_score`         | `float`        | relevance of the article to `symbol`        |
| `summary_embedding`       | `vector(768)`  | embedding of `summary`                      |
| `title_summary_embedding` | `vector(768)`  | embedding of `title` + blank line + `summary` |

Unique key: `(symbol, url)` — `uq_article_key`. The same article can appear
under several tickers, once per ticker. Everything except `id` and the
embeddings may be `NULL`. There is no vector index; similarity search is an
exact scan over one ticker's articles.

### `macro_series`

| Column      | Type          | Notes                                       |
| ----------- | ------------- | ------------------------------------------- |
| `id`        | `integer`     | PK                                          |
| `series_id` | `varchar(32)` | FRED series id (the CSV file name); indexed |
| `date`      | `date`        | indexed                                     |
| `value`     | `float`       |                                             |

Unique key: `(series_id, date)` — `uq_macro_key`. Not tied to a ticker.

## 3. Loading

```bash
uv run python -m src.db.load                                 # everything
uv run python -m src.db.load --datasets fundamentals prices  # chosen datasets
uv run python -m src.db.load --datalake ../Datalake --no-init
uv run python -m src.db.load --no-embed                      # skip embeddings
```

Steps: find `raw_data` → create the schema (`init_db`, unless `--no-init`) →
load the chosen datasets → embed new articles (when `articles` was loaded).

### 3.1 Sources

`resolve_raw_data` looks for a `raw_data` directory in: the `--datalake`
argument, the `DATALAKE_PATH` variable, `/datalake` (the container mount),
`../datalake`, `../Datalake`.

| Dataset (`--datasets`) | Table               | Files in `raw_data/`                                   |
| ---------------------- | ------------------- | ------------------------------------------------------ |
| `fundamentals`         | `fundamentals`      | `BALANCE_SHEET/`, `CASH_FLOW/`, `INCOME_STATEMENT/` — `{TICKER}_{period}.json` with `symbol`, `period`, `reports[]` |
| `prices`               | `stock_prices`      | `stock_price/*.json` — `symbol`, `reports[]` with `date`, `close` |
| `articles`             | `article_summaries` | `ARTICLE_SUMMARIES/{TICKER}_*.json` — a list of articles or `{"feed": [...]}` |
| `macro`                | `macro_series`      | `FRED_MACRO/{SERIES_ID}.csv` — header row, then `DATE,value` |

Only files directly in these directories are read, so subdirectories such as
`deprecated/` are skipped. When `symbol` is missing from a file, it is taken
from the file name prefix (always so for articles).

### 3.2 Cleaning

- Report values (`fundamentals.data`): `"None"`, `""`, `"-"`, `"."` → `null`;
  integer and decimal strings → numbers; other strings unchanged.
  `reportedDate` is moved out of `data` into the `reported_date` column.
- Dates: `YYYY-MM-DD`; invalid dates become `NULL`.
- `time_published`: `20250101T093000`, `20250101T0930` or `YYYY-MM-DD`.
- Price rows without a date or close, macro rows without a date or value, and
  articles without a URL are skipped.

### 3.3 Upsert

Every loader runs `INSERT ... ON CONFLICT (<unique key>) DO UPDATE` in
batches of 500 rows, then commits. Re-running a load updates rows in place
and never duplicates them. Rows deleted from the datalake are not deleted
from the database.

For `article_summaries`, the update sets only the columns read from the file,
so stored embeddings survive a re-load. An embedding is reset to `NULL` when
its source text changed (`summary` for `summary_embedding`; `title` or
`summary` for `title_summary_embedding`), so the next embedding step
recomputes it.

### 3.4 Embeddings

`embed_article_summaries` selects articles with a `NULL` embedding, builds the
text for each column, sends it to Vertex AI `text-embedding-005`
(`src/retriever/embeddings.py`, 768 dims) and commits every 200 rows. An
interrupted run resumes where it stopped: re-run
`python -m src.db.load --datasets articles`.

- Needs `GCP_PROJECT_ID` and Application Default Credentials
  (`gcloud auth application-default login`). Without `GCP_PROJECT_ID` the step
  is skipped.
- Articles with no text get no vector and are never returned by similarity
  search. `count_articles_without_embedding` reports articles that have text
  but no vector yet.

### 3.5 Schema changes

There are no Alembic migrations. `init_db` runs:

1. `CREATE EXTENSION IF NOT EXISTS vector`,
2. `Base.metadata.create_all` — creates missing tables, never alters existing ones,
3. the `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`
   statements in `_ADDED_COLUMNS` for columns added after the first release
   (both embedding columns, `fundamentals.reported_date` and its index).

A new column on an existing table needs an entry in `_ADDED_COLUMNS` as well
as in `models.py`. Changing a column type or a unique key needs a manual
migration or a fresh volume (`docker compose down -v`, then a full load).

## 4. Reading: point-in-time queries

Every read used by a debate is bounded by the debate's `as_of_date`, so the
agents never see data published after the simulated "today"
(`docs/evaluation.md` §1). The bound is **inclusive**
(`docs/project_decisions.md`, 2026-09-27).

| Function (`src/db/queries.py`)     | Used by                         | Point-in-time filter |
| ---------------------------------- | ------------------------------- | -------------------- |
| `fetch_fundamentals`               | Financial agent                 | `coalesce(reported_date, fiscal_date + 90 days) <= as_of_date`; last 40 distinct fiscal dates, all report types and periods for each |
| `fetch_prices`                     | Financial agent                 | `date <= as_of_date`, last 3600 days |
| `fetch_macro_series`               | Macro agent                     | `date <= as_of_date`, last 3600 days, all series |
| `fetch_recent_articles`            | Sentiment agent                 | `time_published < as_of_date + 1 day`; 10 newest |
| `fetch_similar_articles`           | Retriever (`search_articles` tool) | as above, plus cosine similarity ≥ 0.5 on `title_summary_embedding` by default |
| `fetch_articles_by_ids`            | Sentiment agent (later rounds)  | as above, plus the ids found in earlier rounds |
| `count_articles_without_embedding` | Sentiment agent (warning)       | as above |
| `fetch_tickers`                    | `GET /tickers`, `POST /predict/start` | none — not debate data |
| `fetch_price_date_range`           | `GET /meta`                     | none — not debate data |

Two details of the filters:

- **Reports count from their publication date, not their fiscal date.** A
  quarter ending 31 March is published weeks later, so filtering on
  `fiscal_date` alone would leak it into debates dated in between. Rows
  without `reported_date` fall back to `fiscal_date + 90 days`
  (`REPORT_LAG_FALLBACK`), an upper bound on the SEC filing deadline.
- **Timestamps use `< as_of_date + 1 day`.** `time_published` is a timestamp;
  `<= as_of_date` would compare against midnight and drop the articles
  published on `as_of_date` itself.

Article queries never load the embedding columns (`defer`), as callers do not
need them.

## 5. Useful SQL

```sql
-- Row counts and covered dates per ticker
SELECT symbol, count(*), min(date), max(date)
FROM stock_prices GROUP BY symbol ORDER BY symbol;

-- Reports without a publication date (they use the 90-day fallback)
SELECT symbol, report_type, period, count(*)
FROM fundamentals WHERE reported_date IS NULL
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3;

-- A field from the JSONB payload
SELECT symbol, fiscal_date, reported_date, (data->>'totalAssets')::numeric AS total_assets
FROM fundamentals
WHERE report_type = 'BALANCE_SHEET' AND period = 'quarterly' AND symbol = 'AAPL'
ORDER BY fiscal_date DESC;

-- Embedding progress
SELECT count(*) AS articles,
       count(summary_embedding) AS summary_vectors,
       count(title_summary_embedding) AS title_summary_vectors
FROM article_summaries;
```

## 6. Connection

| Setting  | Value                                                       |
| -------- | ----------------------------------------------------------- |
| URL      | `DATABASE_URL`, default `postgresql://user:password@localhost:5432/mas_db` |
| In Docker | `postgresql://user:password@postgres:5432/mas_db` (set in `docker-compose.yml`) |
| Storage  | Docker volume `postgres_data`                               |

`src/db/session.py` creates one engine per process; `get_session()` returns a
new session that the caller closes. Query functions take a session and never
open or commit one themselves.
