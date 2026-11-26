"""Kronos zero-shot baseline for the backtest.

Kronos (Shi et al. 2025, arXiv 2508.02739) is a generative foundation model of
OHLCV candles; its pre-training data ends in June 2024, before the backtest
window.
"""

import argparse
import hashlib
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from scripts.evaluation.io import (
    EVALUATION_DIR,
    append_jsonl,
    compact_predictions,
    read_jsonl,
)
from scripts.evaluation.labels import triple_barrier
from src.agents.volatility import HISTORY_SESSIONS, HORIZON_SESSIONS, horizon_sigma

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "../datalake/raw_data"))
PRICES_CSV = RAW_DATA_DIR / "stock_prices_daily_from_2010.csv"
KRONOS_DIR = Path(os.getenv("KRONOS_DIR", "third_party/Kronos"))

CLASSES = ("BUY", "HOLD", "SELL")
OHLCV = ["open", "high", "low", "close", "volume"]
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# (context bars, context dates, future dates, n_paths, seed) -> closes (n_paths, n)
Sampler = Callable[[pd.DataFrame, pd.Series, pd.Series, int, int], np.ndarray]


# ------------------------------------------------------------------- data --
def load_ohlcv(path: Path = PRICES_CSV) -> dict[str, pd.DataFrame]:
    """Ticker -> daily OHLCV (lower-case columns, DatetimeIndex, oldest first)."""
    raw = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
    raw.index = pd.to_datetime(raw.index)
    out = {}
    for ticker in raw.columns.get_level_values(1).unique():
        df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = OHLCV
        out[ticker] = df.dropna().sort_index()
    return out


def future_sessions(last: pd.Timestamp, n: int) -> pd.Series:
    """Dates of the n sessions after `last`: calendar only, no prices."""
    return pd.Series(pd.date_range(last + BDAY, periods=n, freq=BDAY))


def sample_seed(base: int, ticker: str, as_of: str) -> int:
    """Seed of one sample, independent of the order the samples are run in."""
    digest = hashlib.sha256(f"{base}|{ticker}|{as_of}".encode()).hexdigest()
    return int(digest[:8], 16)


# ------------------------------------------------------------------- core --
def class_probs(
    sampler: Sampler,
    ohlcv: pd.DataFrame,
    as_of: str,
    horizon: str,
    k: float,
    lookback: int,
    n_paths: int,
    seed: int,
) -> dict:
    """P(BUY/HOLD/SELL) of one sample from `n_paths` sampled price paths."""
    # 1. point-in-time: nothing after as_of is used below
    known = ohlcv.loc[ohlcv.index <= pd.Timestamp(as_of)]
    if len(known) < max(lookback, 3):
        raise ValueError(f"only {len(known)} sessions up to {as_of}, need {lookback}")
    hist = known.tail(lookback)

    # 2. barriers exactly as in labels.build_labels
    p0 = float(known["close"].iloc[-1])
    closes_desc = known["close"].tail(HISTORY_SESSIONS).tolist()[::-1]
    sigma_h = horizon_sigma(closes_desc, horizon)
    n = HORIZON_SESSIONS[horizon]

    # 3. model input: context bars, their dates, dates of the n future sessions
    x_df = hist[OHLCV].reset_index(drop=True)
    x_ts = pd.Series(hist.index)
    y_ts = future_sessions(hist.index[-1], n)

    # 4. sampled closes, shape (n_paths, n)
    paths = sampler(x_df, x_ts, y_ts, n_paths, seed)

    # 5. every path scored by the ground-truth rule
    days = [d.date() for d in y_ts]
    votes = Counter(
        triple_barrier(p0, zip(days, path), sigma_h, k)[0] for path in paths
    )
    return {
        "probs": {c: votes[c] / n_paths for c in CLASSES},
        "p0": p0,
        "sigma_h": sigma_h,
        "mean_ret_end": float(paths[:, -1].mean() / p0 - 1),
    }


def decide(probs: dict[str, float]) -> str:
    """Most likely class; a tie goes to HOLD (no call without a clear majority)."""
    return max(CLASSES, key=lambda c: (probs[c], c == "HOLD"))


# ----------------------------------------------------------------- Kronos --
def make_kronos_sampler(
    model_id: str,
    tokenizer_id: str,
    T: float = 1.0,
    top_p: float = 1.0,
    batch: int = 100,
    device: str = "cpu",
) -> Sampler:
    import torch

    sys.path.insert(0, str(KRONOS_DIR.resolve()))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id).eval()
    model = Kronos.from_pretrained(model_id).eval()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    def sample(x_df, x_ts, y_ts, n_paths, seed):
        torch.manual_seed(seed)
        closes: list[np.ndarray] = []
        while len(closes) < n_paths:
            m = min(batch, n_paths - len(closes))
            outs = predictor.predict_batch(
                [x_df] * m,
                [x_ts] * m,
                [y_ts] * m,
                pred_len=len(y_ts),
                T=T,
                top_k=0,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            closes += [o["close"].to_numpy() for o in outs]
        return np.stack(closes)

    return sample


# ------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Kronos zero-shot baseline")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--labels", default=str(EVALUATION_DIR / "labels_1M_k1.jsonl"))
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--lookback", type=int, default=40)  # paper, Table 8 (daily)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = read_jsonl(Path(args.labels))
    ks = {r["k"] for r in manifest}
    if len(ks) != 1:
        raise SystemExit(f"{args.labels}: expected one k, got {ks}")
    k = ks.pop()

    out = EVALUATION_DIR / f"predictions_{args.run_id}.jsonl"

    kept, dropped = compact_predictions(out)
    if dropped:
        print(f"{out.name}: kept {kept} successful rows, dropped {dropped} failed")
    done = {
        (r["ticker"], r["as_of"], r["horizon"])
        for r in read_jsonl(out)
        if r.get("direction")
    }
    todo = [r for r in manifest if (r["ticker"], r["as_of"], r["horizon"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"run {args.run_id}: {len(done)} done, {len(todo)} to run, k={k:g}")
    if not todo:
        return

    ohlcv = load_ohlcv()
    sampler = make_kronos_sampler(
        args.model, args.tokenizer, args.T, args.top_p, args.batch
    )

    for i, r in enumerate(todo, 1):
        base = {"ticker": r["ticker"], "as_of": r["as_of"], "horizon": r["horizon"]}
        t0 = time.perf_counter()
        try:
            res = class_probs(
                sampler,
                ohlcv[r["ticker"]],
                r["as_of"],
                r["horizon"],
                k,
                args.lookback,
                args.paths,
                sample_seed(args.seed, r["ticker"], r["as_of"]),
            )
            # Same inputs as the ground truth? (p0, sigma_h are known at as_of.)
            if not (
                math.isclose(res["p0"], r["p0"], rel_tol=1e-6)
                and math.isclose(res["sigma_h"], r["sigma_h"], rel_tol=1e-6)
            ):
                raise ValueError(
                    f"p0/sigma_h {res['p0']:.4f}/{res['sigma_h']:.5f} differ from "
                    f"the labels file {r['p0']:.4f}/{r['sigma_h']:.5f}"
                )
            probs = res["probs"]
            direction = decide(probs)
            row = {
                **base,
                "direction": direction,
                "confidence": probs[direction],
                "p_buy": probs["BUY"],
                "p_hold": probs["HOLD"],
                "p_sell": probs["SELL"],
                "mean_ret_end": res["mean_ret_end"],
                "error": None,
            }
            info = " ".join(f"{c}={probs[c]:.2f}" for c in CLASSES)
        except Exception as exc:
            row = {**base, "direction": None, "error": repr(exc)}
            info = row["error"]
        append_jsonl(out, row)
        dt = time.perf_counter() - t0
        print(
            f"[{i}/{len(todo)}] {r['ticker']} {r['as_of']} -> "
            f"{row['direction']}  {info}  ({dt:.1f}s)"
        )


if __name__ == "__main__":
    main()
