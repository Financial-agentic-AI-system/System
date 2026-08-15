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
