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
  filled by `python -m src.db.load`, not by a separate transformer script.
