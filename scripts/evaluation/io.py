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


def compact_predictions(path: Path) -> tuple[int, int]:
    """Keep only the successful rows (with a `direction`) in `path`.

    Failed rows and lines that do not parse (e.g. the zero padding left by an
    interrupted write) are dropped; failures are visible in the console while a
    run goes on and the next run retries those samples anyway. The rewrite goes
    through a temp file and os.replace, so a crash cannot lose results.
    Returns (kept, dropped); the file is left alone when it is clean.
    """
    if not path.exists():
        return 0, 0
    kept: list[dict] = []
    dropped = 0
    dirty = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip("\x00 \t"):  # blank line or zero padding
            dirty = True
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if row.get("direction"):
            kept.append(row)
        else:
            dropped += 1
    if not (dropped or dirty):
        return len(kept), 0

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in kept))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(kept), dropped


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
