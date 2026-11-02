"""Backtest evaluation CLI. Three stages, joined only at the end:

  labels   ground truth from prices           (no LLM)
  predict  run the debates for those samples  (LLM; run in backtest-worker)
  metrics  join predictions with labels files, report (no DB, no LLM)

    python -m scripts.evaluation.evaluate labels --horizon 1M --k 0.5 1 1.5 2
    docker compose --profile backtest run --rm backtest-worker \
        uv run --no-sync python -m scripts.evaluation.evaluate predict --run-id r1
    python -m scripts.evaluation.evaluate metrics \
        --predictions ../datalake/evaluation/predictions_r1.jsonl --k 0.5 1 1.5 2

Everything goes to datalake/evaluation (override: EVALUATION_DIR): one
labels_<horizon>_k<k>.jsonl per k, predictions_<run>.jsonl (+ .meta.json) and
metrics_<run>.json. `metrics` needs a labels file for every k it is asked for.
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from scripts.evaluation.io import (
    EVALUATION_DIR,
    append_jsonl,
    compact_predictions,
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
from src.worker.tasks import execute_debate


# ------------------------------------------------------------------ labels --
def labels_path(horizon: str, k: float) -> Path:
    return EVALUATION_DIR / f"labels_{horizon}_k{k:g}.jsonl"


def cmd_labels(args) -> None:
    samples = build_grid(date.fromisoformat(args.start), args.months, args.horizon)
    for k in args.k:
        with get_session() as session:
            labels, skipped = build_labels(session, samples, k, args.span)
        out = labels_path(args.horizon, k)
        write_jsonl(out, labels)
        dist = {
            c: sum(r["label"] == c for r in labels) for c in ("BUY", "HOLD", "SELL")
        }
        print(
            f"k={k:g}: {len(labels)} labels -> {out}   skipped: {len(skipped)}   {dist}"
        )
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
            res = execute_debate(
                sample.ticker, sample.horizon, sample.as_of, task_id=f"eval-{uuid4()}"
            )
            return {
                **base,
                "direction": res.direction,
                "confidence": res.confidence,
                "rounds_used": res.rounds_used,
                "critic_agreed": res.critic_agreed,
                # The PM's round-1 call, before any Critic feedback.
                "first_direction": (
                    res.round_log[0].pm_direction if res.round_log else None
                ),
                "round_log": [r.model_dump() for r in res.round_log],
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
    out = EVALUATION_DIR / f"predictions_{run_id}.jsonl"
    write_meta(
        EVALUATION_DIR / f"predictions_{run_id}.meta.json",
        model_version=llm_client.MODEL_VERSION,
        labels_file=args.labels,
        n_samples=len(samples),
    )

    kept, dropped = compact_predictions(out)
    if dropped:
        print(f"{out.name}: kept {kept} successful rows, dropped {dropped} failed")

    done = {
        (r["ticker"], r["as_of"], r["horizon"])
        for r in read_jsonl(out)
        if r.get("direction")
    }
    todo = [s for s in samples if s.key not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"run {run_id}: {len(done)} done, {len(todo)} to run")

    for i, sample in enumerate(todo, 1):  # sequential on purpose, no threads
        row = run_one(sample)
        append_jsonl(out, row)
        print(
            f"[{i}/{len(todo)}] {row['ticker']} {row['as_of']} -> "
            f"{row['direction']} {row['error'] or ''}"
        )


# ----------------------------------------------------------------- metrics --
def load_labels(horizons: set[str], k: float) -> list[dict]:
    """Labels of every horizon in the predictions, from the labels files."""
    rows: list[dict] = []
    for horizon in sorted(horizons):
        path = labels_path(horizon, k)
        if not path.exists():
            raise SystemExit(
                f"missing {path}; create it with: "
                "python -m scripts.evaluation.evaluate labels "
                f"--horizon {horizon} --k {k:g}"
            )
        rows += read_jsonl(path)
    return rows


def cmd_metrics(args) -> None:
    pred_path = Path(args.predictions)
    preds = read_jsonl(pred_path)
    keys = {(p["ticker"], p["as_of"], p["horizon"]) for p in preds}
    ok = {(p["ticker"], p["as_of"], p["horizon"]) for p in preds if p.get("direction")}
    horizons = {key[2] for key in keys}
    print(f"{len(keys)} samples in file, without a successful call: {len(keys - ok)}")

    summary = []
    for k in args.k:
        labels = load_labels(horizons, k)
        rows = join(preds, labels)
        unlabeled = len(ok) - len(rows)
        if unlabeled:
            print(f"k={k:g}: {unlabeled} predictions have no label (ignored)")
        res = evaluate(rows)
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

    run_id = pred_path.stem.removeprefix("predictions_")
    out = EVALUATION_DIR / f"metrics_{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "predictions_file": str(pred_path),
                "samples": len(keys),
                "without_successful_call": len(keys - ok),
                "results": {f"{k:g}": res for k, res in summary},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved -> {out}")


# -------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("labels")
    p.add_argument("--horizon", default="1M", choices=["1W", "1M", "3M"])
    p.add_argument("--start", default="2025-02-01")
    p.add_argument("--months", type=int, default=19)
    p.add_argument("--k", type=float, nargs="+", default=[1.0])
    p.add_argument("--span", type=int, default=VOL_SPAN)
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("predict")
    p.add_argument("--labels", default=str(labels_path("1M", 1.0)))
    p.add_argument("--run-id")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("metrics")
    p.add_argument("--predictions", required=True)
    p.add_argument("--k", type=float, nargs="+", default=[1.0])
    p.set_defaults(fn=cmd_metrics)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
