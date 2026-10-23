"""Metrics over (prediction, label) pairs. Pure functions, no DB, no LLM."""

from collections import Counter

LABELS = ["BUY", "HOLD", "SELL"]
CONF_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 1.01)]


def _key(r: dict) -> tuple:
    return (r["ticker"], r["as_of"], r["horizon"])


def join(predictions: list[dict], labels: list[dict]) -> list[dict]:
    """Pairs with a successful prediction AND a label, keyed by the sample.

    Later successful predictions override earlier ones (resumed runs).
    """
    preds: dict[tuple, dict] = {}
    for p in predictions:
        if p.get("direction"):
            preds[_key(p)] = p
    lab = {_key(r): r for r in labels}
    return [{**preds[k], "label": lab[k]["label"]} for k in preds if k in lab]


def evaluate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    true_dist = Counter(r["label"] for r in rows)
    pred_dist = Counter(r["direction"] for r in rows)
    correct = sum(r["direction"] == r["label"] for r in rows)

    confusion = {
        t: {
            p: sum(r["label"] == t and r["direction"] == p for r in rows)
            for p in LABELS
        }
        for t in LABELS
    }
    per_class = {}
    for c in LABELS:
        tp = confusion[c][c]
        per_class[c] = {
            "precision": tp / pred_dist[c] if pred_dist[c] else None,
            "recall": tp / true_dist[c] if true_dist[c] else None,
        }

    calibration = []
    for lo, hi in CONF_BINS:
        sub = [r for r in rows if lo <= r["confidence"] < hi]
        calibration.append(
            {
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "n": len(sub),
                "accuracy": (
                    sum(r["direction"] == r["label"] for r in sub) / len(sub)
                    if sub
                    else None
                ),
            }
        )

    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "majority_baseline": max(true_dist.values()) / n,
        "true_dist": dict(true_dist),
        "pred_dist": dict(pred_dist),
        "confusion": confusion,
        "per_class": per_class,
        "calibration": calibration,
    }


def _fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.2f}"


def format_report(res: dict, title: str = "") -> str:
    if res["n"] == 0:
        return f"{title}\nno samples with both a prediction and a label"
    out = [title] if title else []
    out.append(
        f"samples: {res['n']}   correct: {res['correct']} = {res['accuracy']:.1%}   "
        f"majority baseline: {res['majority_baseline']:.1%}"
    )
    out.append(f"true labels: {res['true_dist']}   predictions: {res['pred_dist']}")
    out.append("\nconfusion (rows = true, cols = predicted)")
    out.append("        " + "".join(f"{p:>7}" for p in LABELS))
    for t in LABELS:
        out.append(f"{t:>7} " + "".join(f"{res['confusion'][t][p]:>7}" for p in LABELS))
    out.append("\nclass    precision  recall")
    for c in LABELS:
        pc = res["per_class"][c]
        out.append(f"{c:>5}    {_fmt(pc['precision']):>9}  {_fmt(pc['recall']):>6}")
    out.append("\ncalibration (confidence bin -> accuracy)")
    for b in res["calibration"]:
        out.append(f"  {b['bin']}  n={b['n']:<3}  acc={_fmt(b['accuracy'])}")
    return "\n".join(out)
