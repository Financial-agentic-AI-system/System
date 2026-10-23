"""Backtest evaluation CLI. Three stages, joined only at the end:

  labels   ground truth from prices           (no LLM)
  predict  run the debates for those samples  (LLM; run in backtest-worker)
  metrics  join predictions with labels, report (any k, no re-run of debates)

    python -m scripts.evaluation.evaluate labels --horizon 1M --k 1
    docker compose --profile backtest run --rm backtest-worker \
        uv run --no-sync python -m scripts.evaluation.evaluate predict --run-id r1
    python -m scripts.evaluation.evaluate metrics \
        --predictions results/predictions_r1.jsonl --k 0.5 1 1.5 2
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from scripts.evaluation.io import (
    RESULTS_DIR,
    append_jsonl,
    read_jsonl,
    write_jsonl,
    write_meta,
)
from scripts.evaluation.labels import build_labels
from scripts.evaluation.metrics import evaluate, format_report, join
from scripts.evaluation.samples import Sample, build_grid
from src.agents import llm_client
from src.agents.volatility import VOL_SPAN
from src.db.session import get_session
from src.worker.tasks import _run_debate


# ------------------------------------------------------------------ labels --
def cmd_labels(args) -> None:
    samples = build_grid(date.fromisoformat(args.start), args.months, args.horizon)
    with get_session() as session:
        labels, skipped = build_labels(session, samples, args.k, args.span)
    out = RESULTS_DIR / f"labels_{args.horizon}_k{args.k:g}.jsonl"
    write_jsonl(out, labels)
    dist = {c: sum(r["label"] == c for r in labels) for c in ("BUY", "HOLD", "SELL")}
    print(f"{len(labels)} labels -> {out}   skipped: {len(skipped)}   {dist}")
    for s in skipped:
        print("  skipped", s)


# ----------------------------------------------------------------- predict --
def run_one(sample: Sample, attempts: int = 3) -> dict:
    base = {
        "ticker": sample.ticker,
        "as_of": sample.as_of.isoformat(),
        "horizon": sample.horizon,
    }
    for attempt in range(attempts):  # the proxy is occasionally flaky
        try:
            res = _run_debate(
                sample.ticker, sample.horizon, sample.as_of, task_id=f"eval-{uuid4()}"
            )
            return {
                **base,
                "direction": res.direction,
                "confidence": res.confidence,
                "rounds_used": res.rounds_used,
                "critic_agreed": res.critic_agreed,
                "error": None,
            }
        except Exception as exc:
            if attempt == attempts - 1:
                return {**base, "direction": None, "error": repr(exc)}


def cmd_predict(args) -> None:
    # The labels file is only a manifest of which samples are labelable; the
    # label values are never passed into the debate.
    samples = [
        Sample(r["ticker"], date.fromisoformat(r["as_of"]), r["horizon"])
        for r in read_jsonl(Path(args.labels))
    ]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M")
    out = RESULTS_DIR / f"predictions_{run_id}.jsonl"
    write_meta(
        RESULTS_DIR / f"predictions_{run_id}.meta.json",
        model_version=llm_client.MODEL_VERSION,
        labels_file=args.labels,
        n_samples=len(samples),
    )

    done = {
        (r["ticker"], r["as_of"], r["horizon"])
        for r in read_jsonl(out)
        if r.get("direction")
    }
    todo = [s for s in samples if s.key not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"run {run_id}: {len(done)} done, {len(todo)} to run")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, todo), 1):
            append_jsonl(out, row)
            print(
                f"[{i}/{len(todo)}] {row['ticker']} {row['as_of']} -> "
                f"{row['direction']} {row['error'] or ''}"
            )


# ----------------------------------------------------------------- metrics --
def cmd_metrics(args) -> None:
    preds = read_jsonl(Path(args.predictions))
    samples = {
        (p["ticker"], p["as_of"], p["horizon"]): Sample(
            p["ticker"], date.fromisoformat(p["as_of"]), p["horizon"]
        )
        for p in preds
    }
    failed = sum(1 for p in preds if not p.get("direction"))
    print(f"{len(samples)} samples in file, failed calls: {failed}")

    summary = []
    for k in args.k:
        with get_session() as session:
            labels, _ = build_labels(session, list(samples.values()), k, args.span)
        res = evaluate(join(preds, labels))
        print("\n" + format_report(res, title=f"=== k = {k:g} ==="))
        summary.append((k, res))

    if len(summary) > 1:
        print("\nsweep:   k   accuracy  baseline  true labels")
        for k, r in summary:
            if r["n"]:
                print(
                    f"       {k:>4g}   {r['accuracy']:>7.1%}  "
                    f"{r['majority_baseline']:>7.1%}  {r['true_dist']}"
                )


# -------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("labels")
    p.add_argument("--horizon", default="1M", choices=["1W", "1M", "3M"])
    p.add_argument("--start", default="2025-02-01")
    p.add_argument("--months", type=int, default=9)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--span", type=int, default=VOL_SPAN)
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("predict")
    p.add_argument("--labels", default="results/labels_1M_k1.jsonl")
    p.add_argument("--run-id")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("metrics")
    p.add_argument("--predictions", required=True)
    p.add_argument("--k", type=float, nargs="+", default=[1.0])
    p.add_argument("--span", type=int, default=VOL_SPAN)
    p.set_defaults(fn=cmd_metrics)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
