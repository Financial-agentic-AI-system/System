# Project Decisions

Short decision log. See `docs/adr/` for longer-form records if this grows
past a few entries.

## 2026-08-15 — LLM: Llama 3.3 70B Instruct (MaaS) instead of Gemini

Chosen over any Gemini model for two reasons:

1. **Training cutoff.** Backtest window starts 2025-01-01 (`docs/evaluation.md`
   §3.3), so the model's cutoff must fall strictly before that. Llama 3.3's
   cutoff is December 2023 — clean margin. Gemini versions with a similarly
   clean pre-2025 cutoff (1.5, 2.0) are deprecated/shut down; the Gemini
   versions still available (2.5+) have a January 2025 cutoff that overlaps
   the window instead of preceding it.
2. **Price.** ~$1.36 per 1M tokens (blended) vs. Gemini 2.5 Pro's $1.25
   input / $10.00 output per 1M — Llama is substantially cheaper at the
   comparable capability tier. (Gemini 2.5 Flash-Lite is nominally cheaper
   still, but is excluded by reason 1.)

Both served through the same platform (Gemini Enterprise Agent Platform /
Vertex AI), same auth, same project — see `src/agents/llm_client.py`.

## 2026-09-27 — `as_of_date` is an inclusive cutoff

Agents see data up to **and including** `as_of_date`, on all four sources.
Chosen over an exclusive bound so the sentiment agent gets that day's news,
which is the most decision-relevant. No lookahead bias either way — nothing
published after `as_of_date` is visible.

## 2026-10-02 — Embeddings: `text-embedding-005`, two columns on `article_summaries`

- **Model.** English-specialised (the articles are English), 768 dims, and
  batched requests. `gemini-embedding-001` takes one text per request, so
  loading the articles would be far slower; it stays available through
  `EMBEDDING_MODEL`.
- **Training cutoff is not a lookahead risk here.** The embedding model
  generates no text, it only ranks stored articles, and retrieval is filtered
  by `as_of_date`.
- **Storage.** `summary_embedding` and `title_summary_embedding` are stored
  side by side so the retriever can compare the two variants. They are
  filled by `python -m src.db.load`.
- **Search defaults: `title_summary`, similarity ≥ 0.5.** On AAPL articles an
  unrelated control query scored at most 0.42, relevant queries' 10th hit at
  least 0.51; `title_summary` separated the two slightly better. Re-check with
  `notebooks/similarity_threshold.ipynb`.

## 2026-10-14 — Sentiment agent: the model asks for a search, the code runs it

- **Tool calling, one call, a list of queries.** The agent first sees the 10
  most recent articles and may call `search_articles` with up to 3 queries
  (`generate_tool_calls`); it can also decline. The smoke test against the MaaS
  endpoint returned well-formed `tool_calls`. A list in one call, instead of a
  loop of turns, keeps it two LLM calls per round: one to choose queries, one
  (the usual `generate_structured`) for the opinion.
- **The model names queries, nothing else.** `ticker` and `as_of_date` are not
  in the tool's schema; the code takes them from the state, so the
  point-in-time bound cannot be argued away. The schema is a Pydantic model
  (`src/agents/tools.py`) that also validates the arguments.
- **Every round, not only the first.** On a re-ask the agent sees the Critic's
  feedback and the queries it already used, and re-reads the articles found in
  earlier rounds (by id, from the state).
- **State accumulates, history shows the round.** `sentiment_queries` and
  `sentiment_sources` use `operator.add` reducers: a node returns only its own
  round's values, so the history delta (Redis, 24 h) is per-round while the
  state is cumulative. Queries and found articles are shown on the frontend;
  they are not part of `PredictResult`.
- **Failures.** A model that skips the tool or returns broken arguments falls
  back to the recent articles. An embedding or SQL error fails the debate: a run
  that silently skipped retrieval would look valid yet not be comparable.
- **Ablation switch.** `SENTIMENT_RETRIEVAL=0` leaves out the search entirely
  (docs/evaluation.md §4). At most 3 queries per round, 5 articles per query,
  on top of the retriever's 0.5 similarity threshold.
