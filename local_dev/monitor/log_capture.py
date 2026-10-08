"""Extract miner request/finish events from pm2 log files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_RECEIVED = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*Miner received task of type (?P<type>\S+)"
)
_FINISHED = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*CompetitiveMiner: finished (?P<type>\S+) "
    r"in (?P<elapsed>[0-9.]+)s(?: \(budget=(?P<budget>[0-9.]+)s (?P<detail>[^)]+)\))?(?P<over> OVER_BUDGET)?"
)
_SAVED = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*Saved miner problem → (?P<path>\S+)"
)
_ERROR = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*(ERROR|Error|Exception|Traceback)"
)


def _tail_lines(path: Path, max_bytes: int = 2_000_000) -> list[str]:
    if not path.is_file():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # drop partial
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines()


def capture_pm2_events(
    out_log: Path,
    err_log: Path | None = None,
    *,
    max_events: int = 200,
) -> dict[str, Any]:
    lines = _tail_lines(out_log)
    if err_log:
        lines += _tail_lines(err_log, max_bytes=500_000)

    received: list[dict] = []
    finished: list[dict] = []
    saved: list[dict] = []
    errors: list[dict] = []

    for line in lines:
        m = _RECEIVED.search(line)
        if m:
            received.append({"ts": m.group("ts"), "task_type": m.group("type")})
            continue
        m = _FINISHED.search(line)
        if m:
            finished.append(
                {
                    "ts": m.group("ts"),
                    "task_type": m.group("type"),
                    "elapsed_s": float(m.group("elapsed")),
                    "budget_s": float(m.group("budget")) if m.group("budget") else None,
                    "detail": m.group("detail"),
                    "over_budget": bool(m.group("over")),
                }
            )
            continue
        m = _SAVED.search(line)
        if m:
            saved.append({"ts": m.group("ts"), "path": m.group("path")})
            continue
        if "ERROR" in line or "Error extracting" in line or "Traceback" in line:
            m = _ERROR.search(line)
            errors.append(
                {
                    "ts": (m.group("ts") if m else None),
                    "line": line[-400:],
                }
            )

    # Keep newest
    received = received[-max_events:]
    finished = finished[-max_events:]
    saved = saved[-max_events:]
    errors = errors[-50:]

    over = [f for f in finished if f.get("over_budget") or (
        f.get("budget_s") and f["elapsed_s"] > f["budget_s"]
    )]
    by_type: dict[str, list[float]] = {}
    for f in finished:
        by_type.setdefault(f["task_type"], []).append(f["elapsed_s"])
    latency = {
        t: {
            "n": len(xs),
            "mean": sum(xs) / len(xs),
            "max": max(xs),
            "min": min(xs),
        }
        for t, xs in by_type.items()
    }

    return {
        "out_log": str(out_log),
        "err_log": str(err_log) if err_log else None,
        "received": received,
        "finished": finished,
        "saved": saved,
        "errors": errors,
        "over_budget": over,
        "latency_by_type": latency,
        "counts": {
            "received": len(received),
            "finished": len(finished),
            "saved": len(saved),
            "errors": len(errors),
            "over_budget": len(over),
        },
    }
