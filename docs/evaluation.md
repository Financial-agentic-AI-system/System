# Evaluation Methodology

This defines how the thesis measures whether the multi-agent debate produces
useful `BUY | HOLD | SELL` calls, via backtesting against historical data.

## 1. Core risk: lookahead bias

A backtest is only meaningful if, at simulated date `D`, the system truly has
access to no information from after `D`. There are **three independent
leakage channels** — the design closes all three, not just one:

1. **Live network access** — the agent calling out to the internet during
   inference and observing today's real price/news.
2. **The LLM's own pretraining knowledge** — a model already "knows" what
   happened to a ticker after `D` from its training data, even with zero
   internet access at inference time. This is a well-known problem in
   LLM-based trading backtests and is separate from (1).
3. **Point-in-time correctness of *your own* database** — `fundamentals`,
   `article_summaries`, `stock_prices`, `macro_series` all contain data
   spanning the full history. If a backtest query for `D = 2025-06-01` isn't
   filtered to what was known on `D`, the agent sees the future through your
   own Postgres instance, independent of any internet restriction.

## 2. Design decisions

- **Network isolation via a proxy container** closes channel (1). The
  worker/agent container has no direct route to the internet; all
  LLM/embedding calls are forced through an allowlisted proxy in a separate
  container. This is a network-layer control, not a prompt instruction, so it
  holds even if the model ignores its system prompt.
- **Point-in-time filtering in the database** closes channel (3), and is the
  DB-side counterpart the proxy cannot provide by itself — the proxy only
  stops the agent from reaching the open internet, it does nothing about
  data the debate reads from your own Postgres. Every query the agents make
  passes an `as_of_date` and filters on what was known by then. See §3.1.
- **Model training cutoff predates the entire backtest window**, not just
  each individual prediction date, closes channel (2). With the backtest
  window starting 2025-01-01 (§3.3), the chosen model's cutoff must fall
  strictly before 2025-01-01 — otherwise predictions made early in the window
  would be clean but later ones would leak via channel (2) regardless of
  network isolation.
- **Grounding disabled at the LLM client config level, not just the network
  level.** If Vertex AI/Gemini's "Grounding with Google Search" tool stays
  enabled on the client, tool calls go out to the same `*.googleapis.com`
  domain the proxy already allowlists for ordinary inference — the network
  control would silently do nothing, since grounding isn't a separate host.
  `src/agents/llm_client.py` must not attach a `google_search` /
  `google_search_retrieval` tool to any call used during backtests.
  Verification step: inspect the outgoing request payload during a backtest
  run and confirm no search/grounding tool is attached. The one tool that
  *is* sent is our own local function `search_articles` (`generate_tool_calls`,
  Sentiment agent only): the model only names its queries, and the code runs
  them against the point-in-time database. So the check is "the only entry in
  `tools` is `search_articles`", not "there is no `tools` field".
- **No wall-clock leakage in prompts.** Any prompt/context builder that
  touches "today" or "current date" (e.g. a macro agent prompt) must source
  that value from the simulated `as_of_date`, never from `datetime.now()`.
  This applies everywhere a date is interpolated into agent input, not just
  the top-level request.
- **Ground-truth labels live outside the database the agents read.** Labels
  use prices from *after* `as_of`, so they are written to files
  (`datalake/evaluation`, §4.3), never to a Postgres table, and are never
  passed into a debate.

## 3. Architecture

### 3.1 Point-in-time retrieval

Every query the agents make (`src/db/queries.py`, the only place with agent SQL)
takes an `as_of_date` and filters on the moment the data became public:

| Table | Filter |
| --- | --- |
| `fundamentals` | `coalesce(reported_date, fiscal_date + 90 days) <= as_of_date` |
| `stock_prices`, `macro_series` | `date <= as_of_date` |
| `article_summaries` | `time_published < as_of_date + 1 day` |

Details:

- **Fundamentals are filtered by publication date, not by fiscal date.** A
  quarter ending 31 March is reported weeks later, so `fiscal_date <= as_of`
  would show it too early. `reported_date` comes from Alpha Vantage `EARNINGS`
  (`reportedDate`) and is loaded by `src/db/loaders.py`. Reports without it
  (all of them older than 2021 in the current datalake) are treated as published
  90 days after the fiscal date. No report from 2021 onwards uses that fallback,
  so it does not affect the backtest.
- **Timestamps use `< as_of_date + 1 day`**, because `time_published` is a
  timestamp and `<= as_of_date` would drop articles published on `as_of_date`.
- **Known remaining gap — macro series.** FRED dates are period starts (a
  monthly CPI value dated 1 June is published in July). `macro_series` is
  filtered by that date, so a value can be visible earlier than it was really
  published. This is not corrected yet and is a limitation of the results.

Details and the full list of queries: `docs/database.md` §4.

### 3.2 Network isolation container

Implemented in `docker-compose.yml` (`backtest` profile) and
`docker/backtest-proxy/squid.conf`. A dedicated `backtest-proxy` service
(squid — chosen over tinyproxy: allowlisting HTTPS `CONNECT` tunnels by
destination domain is squid's standard `acl dstdomain` use case, no SSL
bump needed) sits between the worker container and the internet:

```yaml
backtest-worker:
  # same image as `worker`, used only for backtest runs
  environment:
    - HTTP_PROXY=http://backtest-proxy:8888
    - HTTPS_PROXY=http://backtest-proxy:8888
  networks:
    - backtest-net       # ONLY this network — no default internet route

backtest-proxy:
  image: ubuntu/squid:latest
  volumes:
    - ./docker/backtest-proxy/squid.conf:/etc/squid/squid.conf:ro
  networks:
    - backtest-net
    - default            # only this service can reach the internet
```

Postgres and Redis are also attached to `backtest-net` (in addition to
`default`) so `backtest-worker` — which has no other network — can still
reach the DB and cache; neither needs internet access itself, so this
doesn't weaken the isolation.

Allowlist in `docker/backtest-proxy/squid.conf` is as narrow as the
LLM/embedding calls actually require — currently
`us-east5-aiplatform.googleapis.com` (the inference calls themselves)
and `oauth2.googleapis.com` (ADC token refresh talks to a *different* host
than the inference endpoint — easy to miss, discovered by allowing all +
reading squid's `access.log` during one real call, then locking down).
Narrower than a wildcard on `googleapis.com` is deliberate, since Google
hosts many products (including Search) behind that domain family. The region
in the allowlist must match `GCP_LOCATION` in `.env` (a mismatch shows up as a
403 from the proxy).

Verification: from `backtest-worker`, a direct `curl` to any non-Google
host must fail with no route (proves the network has no default gateway);
a `curl -x http://backtest-proxy:8888` to an allowlisted host must
succeed; the same through the proxy to a non-allowlisted host must get a
squid 403 (proves the allowlist itself is enforced, not just "a proxy
exists"). Run via `docker compose --profile backtest run --rm
backtest-worker ...`.

### 3.3 Backtest window and model cutoff

| Field | Value |
| --- | --- |
| Data available | prices, articles and macro up to 2026-09-30 (fundamentals up to the reports published by then) |
| Backtest window | 2025-01-01 → 2026-09-30 |
| Sample grid | `as_of` = first day of each month, 2025-02 → 2026-08 (19 months); the horizon of the last date still ends inside the data |
| Model + version | Llama 4 Scout 17B-16E Instruct — `meta/llama-4-scout-17b-16e-instruct-maas`, served as MaaS on Vertex AI (`src/agents/llm_client.py`) |
| Required model training cutoff | before 2025-01-01 — satisfied: Meta's official Llama 4 Scout model card lists a training data cutoff of **August 2024** |

Record the model, window and prompts together per backtest run, not
independently: `predictions_<run_id>.meta.json` stores the model version, a hash
of all prompt files and the labels file used (§4.3), and every prediction row
carries its own `as_of` date.

## 4. Procedure

The evaluation is three separate stages that meet only at the end, so the
labels can be recomputed with any `k` without running a single debate again.

```
labels   ground truth from prices           no LLM, runs on the host
predict  the debates for those samples      LLM, runs in backtest-worker
metrics  prediction x label -> numbers      no LLM, runs on the host
```

Code: `scripts/evaluation/` (`evaluate.py` is the CLI). The volatility rule is
shared with the agents in `src/agents/volatility.py`.

### 4.1 What counts as BUY / HOLD / SELL (Triple Barrier)

Labels follow the Triple Barrier Method (López de Prado, *Advances in Financial
Machine Learning*, ch. 3), checked on daily closes after `as_of`:

- **Horizontal barriers** at `p0 * (1 ± k * sigma_h)`, where `p0` is the last
  close on or before `as_of`. The first barrier touched within the horizon
  decides: upper → `BUY`, lower → `SELL`.
- **Vertical barrier** at the end of the horizon (`n` trading sessions after
  `as_of`: 5 for 1W, 21 for 1M, 63 for 3M). Neither horizontal barrier touched
  → `HOLD`.
- **Barrier width:** `sigma_h = ewm_std(daily returns, span 100) * sqrt(n)`,
  computed from the last 250 sessions up to `as_of` (the EWM weights of older
  sessions are negligible). The `sqrt(n)` scaling assumes independent daily
  returns, so the std of an `n`-day return is `sqrt(n)` times the daily one.
- **`k`** is the number of standard deviations. The main result uses `k = 1`;
  `labels` and `metrics` accept several values to show how the conclusion
  depends on it.

The Portfolio Manager prompt states this definition (the horizon, `sigma_h` as a
percentage, and what BUY / HOLD / SELL mean), computed by the same
`horizon_sigma` function, so the model is judged by the rule it was told.

### 4.2 Samples

One sample is one `(ticker, as_of, horizon)`. The grid is 10 tickers (AAPL, CAT,
GS, INTC, META, NFLX, NVDA, PLTR, SMCI, TSLA) × 19 monthly dates = **190
samples** for horizon 1M (`scripts/evaluation/samples.py`). A sample gets no
label, and is listed as skipped with a reason, when there is no price on or before
`as_of`, the horizon runs past the available prices, or there is too little
history to estimate the volatility.

Samples of one ticker within the same quarter share the same fundamentals and
overlap in time, so they are correlated: 190 debates are fewer than 190
independent observations. Confidence intervals (planned, §5) have to account
for that.

### 4.3 Running it

Labels and predictions are files in the datalake repository, next to
`raw_data`, not rows in Postgres:

```
datalake/evaluation/
  labels_1M_k<k>.jsonl           one file per k; one row per sample: p0, sigma_h, barriers, label, touch_date, ret_end
  predictions_<run_id>.jsonl     one row per debate: direction, confidence, rounds, error
  predictions_<run_id>.meta.json model version, prompts hash, labels file, sample count
  metrics_<run_id>.json          result of `metrics` for every k (accuracy, baseline, confusion, calibration)
```

The folder is set by `EVALUATION_DIR` (default `../datalake/evaluation`; in the
container `/datalake/evaluation`, mounted writable inside the otherwise
read-only `/datalake`).

```bash
# 1. labels (host, needs Postgres with prices); one file per k
docker compose up -d postgres redis
uv run python -m scripts.evaluation.evaluate labels --horizon 1M --k 0.5 1 1.5 2

# 2. predictions (container behind the proxy; resumable, see below)
docker compose --profile backtest build backtest-worker   # after any change in src/
docker compose --profile backtest run --rm backtest-worker \
    uv run --no-sync python -m scripts.evaluation.evaluate predict --run-id r1 [--limit 5]

# 3. metrics (host; reads the files, no database)
uv run python -m scripts.evaluation.evaluate metrics \
    --predictions ../datalake/evaluation/predictions_r1.jsonl --k 0.5 1 1.5 2
```

Notes:

- `src/` is baked into the worker image, so rebuild it after changing code;
  `scripts/` is mounted and picked up without a rebuild.
- `predict` reads the labels file only as a list of samples; the label values
  are never given to a debate. A run is resumable: samples with a prediction are
  skipped, failed ones (rows with `error`) are retried on the next call. Each
  sample is tried 3 times.
- The proxy must be running with an allowlist matching the region in `.env`
  (§3.2). Start with `--limit 5` to check the whole path before a full run.
- `metrics` reads `labels_<horizon>_k<k>.jsonl` for every `k` it is asked for
  and does not touch the database, so the result depends only on the two files
  and can be recomputed anywhere. A missing labels file stops it with the
  command that creates it. Predictions without a label are counted and ignored.
  The result is printed and saved to `metrics_<run_id>.json`.

### 4.4 Reproducibility and run-to-run variance

The LLM runs at temperature 0, but a hosted model is not guaranteed to return
the same text twice, and a debate has several rounds that can diverge. A single
run is therefore one draw. The thesis reports the exact model, prompt hash and
configuration per run, and repeats the full run to show how much the result
moves. Configuration (model, prompts, `k`, span, retrieval settings) is frozen
before the final run so that no setting is tuned on the evaluation samples.

## 5. Baselines and metrics

A `BUY/HOLD/SELL` signal needs a comparison point, not just raw accuracy.

**Implemented** (`scripts/evaluation/metrics.py`, reported for each `k`):

- accuracy and the **majority-class baseline** (always predicting the most
  frequent true label) — `HOLD` dominates the label distribution, so accuracy
  alone is misleading;
- confusion matrix, precision and recall per class;
- **confidence calibration:** predictions bucketed by the PM's `confidence`
  (0–0.5, 0.5–0.7, 0.7–1), with accuracy per bucket, to check whether
  high-confidence calls are actually more often correct.


## 6. Known limitations

- **Macro publication lag** is not modelled (§3.1).
- **Retrieval has no recency preference.** `search_articles` ranks by cosine
  similarity only (threshold 0.5); the number of articles also varies strongly
  over time (roughly 80–440 a month across all tickers until September 2025,
  several thousand a month in October–November 2025), so the quality of the
  sentiment input depends on the date.
- **Quarterly fundamentals** change once per quarter, so monthly samples of the
  same ticker often show the Financial agent the same report (§4.2).
- **Non-determinism** of the hosted LLM (§4.4).
