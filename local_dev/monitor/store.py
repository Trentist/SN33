"""Persist monitor snapshots under local_dev/cache/monitor/uidN/."""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "snapshots": out_dir / "snapshots",
        "latest": out_dir / "latest",
        "scores": out_dir / "scores",
        "logs": out_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def save_snapshot(
    out_dir: Path,
    *,
    scores: dict[str, Any],
    logs: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    dirs = ensure_dirs(out_dir)
    stamp = _utc_stamp()
    now = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "saved_at": now,
        "stamp": stamp,
        "meta": meta or {},
        "scores": scores,
        "logs": logs,
    }

    snap_path = dirs["snapshots"] / f"{stamp}.json"
    _write_json(snap_path, snapshot)

    # Rolling "latest" pointers for easy tailing
    _write_json(dirs["latest"] / "snapshot.json", snapshot)
    _write_json(dirs["latest"] / "scores.json", scores)
    _write_json(dirs["latest"] / "logs.json", logs)

    # Append-only score history (one line per poll with aggregate + newest match)
    hist_path = dirs["scores"] / "history.jsonl"
    newest = (scores.get("matches") or [None])[0]
    hist_row = {
        "saved_at": now,
        "aggregate": scores.get("aggregate"),
        "matches_n": len(scores.get("matches") or []),
        "newest": newest,
    }
    with hist_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(hist_row, default=str) + "\n")

    # Compact latency history
    lat_path = dirs["logs"] / "latency_history.jsonl"
    with lat_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "saved_at": now,
                    "counts": logs.get("counts"),
                    "latency_by_type": logs.get("latency_by_type"),
                    "over_budget_n": len(logs.get("over_budget") or []),
                    "last_finished": (logs.get("finished") or [None])[-1],
                },
                default=str,
            )
            + "\n"
        )

    # Keep only last 200 snapshots
    snaps = sorted(dirs["snapshots"].glob("*.json"))
    for old in snaps[:-200]:
        try:
            old.unlink()
        except Exception:
            pass

    return {
        "snapshot": str(snap_path),
        "latest": str(dirs["latest"] / "snapshot.json"),
        "scores_history": str(hist_path),
        "latency_history": str(lat_path),
    }
