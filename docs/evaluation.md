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
   filtered to `date <= D`, the agent sees the future through your own
   Postgres instance, independent of any internet restriction.

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
  passes an `as_of_date` and filters `WHERE date <= as_of_date` (or
  `fiscal_date` / `time_published`, per table) across `fundamentals`,
  `stock_prices`, `macro_series`, `article_summaries`. See §3.1.
- **Model training cutoff predates the entire backtest window**, not just
  each individual prediction date, closes channel (2). With the backtest
  window fixed at 2025-01-01 → mid-2026 (§3.3), the chosen model's cutoff
  must fall strictly before 2025-01-01 — otherwise predictions made early in
  the window would be clean but later ones would leak via channel (2)
  regardless of network isolation.
- **Grounding disabled at the LLM client config level, not just the network
  level.** If Vertex AI/Gemini's "Grounding with Google Search" tool stays
  enabled on the client, tool calls go out to the same `*.googleapis.com`
  domain the proxy already allowlists for ordinary inference — the network
  control would silently do nothing, since grounding isn't a separate host.
  `src/agents/llm_client.py` must not attach a `google_search` /
  `google_search_retrieval` tool to any call used during backtests.
  Verification step: inspect the outgoing request payload during a backtest
  run and confirm no search/grounding tool is attached.
- **No wall-clock leakage in prompts.** Any prompt/context builder that
  touches "today" or "current date" (e.g. a macro agent prompt) must source
  that value from the simulated `as_of_date`, never from `datetime.now()`.
  This applies everywhere a date is interpolated into agent input, not just
  the top-level request.

## 3. Architecture

### 3.1 Point-in-time retrieval

Every query the agents make (via `src/retriever` and direct SQL in
`src/agents/nodes.py`) must accept an `as_of_date` parameter and filter:

```sql
-- fundamentals / stock_prices / macro_series / article_summaries
WHERE fiscal_date <= :as_of_date   -- or date / time_published, per table
```

### 3.2 Network isolation container

A dedicated `backtest-proxy` service (e.g. `tinyproxy` or `squid`), sitting
between the worker container and the internet:

```yaml
backtest-worker:
  # same image as `worker`, used only for backtest runs
  environment:
    - HTTP_PROXY=http://backtest-proxy:8888
    - HTTPS_PROXY=http://backtest-proxy:8888
  networks:
    - backtest-net       # no default internet route on this network

backtest-proxy:
  image: tinyproxy
  volumes:
    - ./docker/backtest-proxy.conf:/etc/tinyproxy/tinyproxy.conf:ro
  networks:
    - backtest-net
    - default            # only this service can reach the internet
```

Allowlist in `backtest-proxy.conf` should be as narrow as the LLM/embedding
calls actually require (typically the specific regional Vertex AI endpoint,
e.g. `us-central1-aiplatform.googleapis.com`, rather than a wildcard on
`googleapis.com`) — narrower is better since Google hosts many products
(including Search) behind that domain family.

### 3.3 Backtest window and model cutoff

| Field | Value |
| --- | --- |
| Backtest window | 2025-01-01 → mid-2026 |
| Model + version | Llama 3.3 70B Instruct — `meta/llama-3.3-70b-instruct-maas`, served as MaaS on Gemini Enterprise Agent Platform (`src/agents/llm_client.py`) |
| Required model training cutoff | before 2025-01-01 — satisfied: Meta's official Llama 3.3 model card lists a training data cutoff of **December 2023** |

Record this triple together per backtest run, not independently — it's
already captured per-prediction in the `model_version` / `as_of_date` fields
of the prediction contract (`docs/ARCHITECTURE.md` §5), so results stay
reproducible and auditable after the fact.



## 4. Baselines and metrics

A `BUY/HOLD/SELL` signal needs a comparison point, not just raw accuracy:

- **Baselines:** buy-and-hold, naive momentum (sign of last N-day return),
  random signal (as a sanity floor).
- **Directional metrics:** hit rate (% correct direction), precision/recall
  per class — `HOLD` will likely dominate the label distribution, so plain
  accuracy alone will be misleading.
- **Financial metrics:** cumulative return of a simulated strategy that
  follows the signals vs. the baselines above; max drawdown.
- **Confidence calibration:** bucket predictions by `confidence` and check
  whether high-confidence calls are actually more often correct — validates
  whether the PM's self-reported confidence is meaningful at all.
- **Ablation:** re-run the debate with one worker agent silenced at a time
  (Financial-only, Sentiment-only, Macro-only, no-Critic) to show which
  agents actually move the outcome — useful evidence for the thesis that the
  multi-agent structure adds value over a single LLM call.

