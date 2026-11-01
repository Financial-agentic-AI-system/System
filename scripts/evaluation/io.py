"""JSONL / meta.json helpers for the evaluation folder (files, not the DB)."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Labels and predictions live in the datalake repo (datalake/evaluation), next to
# raw_data. Inside the backtest container it is mounted at /datalake/evaluation.
EVALUATION_DIR = Path(os.getenv("EVALUATION_DIR", "../datalake/evaluation"))
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "src" / "agents" / "prompts"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def prompts_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PROMPTS_DIR.glob("*.yaml")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def write_meta(path: Path, **config) -> None:
    """Everything needed to say later which model/prompt produced a run."""
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts_hash": prompts_hash(),
        **config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
