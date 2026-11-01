"""`evaluate metrics` reads labels files (no DB) and saves a metrics JSON."""

import json

import pytest

from scripts.evaluation import evaluate as ev
from scripts.evaluation.io import read_jsonl, write_jsonl


def _row(ticker, label=None, direction=None, **extra):
    row = {"ticker": ticker, "as_of": "2025-02-01", "horizon": "1M", **extra}
    if label:
        row["label"] = label
    if direction is not None or "error" in extra:
        row["direction"] = direction
    return row


class Args:
    def __init__(self, predictions, k):
        self.predictions = str(predictions)
        self.k = k


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVALUATION_DIR", tmp_path)
    return tmp_path


def test_metrics_uses_labels_file_and_saves_result(folder, capsys):
    write_jsonl(
        folder / "labels_1M_k1.jsonl",
        [_row("A", "BUY"), _row("B", "HOLD"), _row("C", "SELL")],
    )
    write_jsonl(
        folder / "predictions_r1.jsonl",
        [
            _row("A", direction=None, error="boom"),  # failed attempt, retried
            _row("A", direction="BUY", confidence=0.8),
            _row("B", direction="BUY", confidence=0.6),
            _row("D", direction="SELL", confidence=0.7),  # no label for D
            _row("E", direction=None, error="boom"),  # never succeeded
        ],
    )

    ev.cmd_metrics(Args(folder / "predictions_r1.jsonl", [1.0]))

    out = capsys.readouterr().out
    assert "without a successful call: 1" in out
    assert "1 predictions have no label" in out
    saved = json.loads((folder / "metrics_r1.json").read_text(encoding="utf-8"))
    assert saved["samples"] == 4
    assert saved["without_successful_call"] == 1
    assert saved["results"]["1"]["n"] == 2
    assert saved["results"]["1"]["correct"] == 1


def test_metrics_names_the_missing_labels_file(folder):
    write_jsonl(
        folder / "predictions_r1.jsonl", [_row("A", direction="BUY", confidence=0.8)]
    )

    with pytest.raises(SystemExit) as exc:
        ev.cmd_metrics(Args(folder / "predictions_r1.jsonl", [2.0]))

    assert "labels_1M_k2.jsonl" in str(exc.value)
    assert "--k 2" in str(exc.value)


def test_labels_path_includes_horizon_and_k(folder):
    assert ev.labels_path("3M", 0.5) == folder / "labels_3M_k0.5.jsonl"
    assert read_jsonl(folder / "nothing.jsonl") == []
