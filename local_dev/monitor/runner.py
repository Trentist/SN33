"""One-shot and loop runners for the SN33 UID monitor."""

from __future__ import annotations

import json
import time
import traceback
from typing import Any

from local_dev.monitor.config import MonitorConfig
from local_dev.monitor.config import default_config
from local_dev.monitor.log_capture import capture_pm2_events
from local_dev.monitor.store import save_snapshot
from local_dev.monitor.wandb_scores import fetch_uid_scores


def run_once(cfg: MonitorConfig | None = None) -> dict[str, Any]:
    cfg = cfg or default_config()
    errors: list[str] = []

    scores: dict[str, Any] = {}
    try:
        scores = fetch_uid_scores(
            uid=cfg.uid,
            hotkey=cfg.hotkey,
            entity=cfg.entity,
            project=cfg.project,
            max_runs=cfg.wandb_runs,
        )
    except Exception as e:
        errors.append(f"wandb: {e}")
        scores = {
            "uid": cfg.uid,
            "hotkey": cfg.hotkey,
            "error": str(e),
            "matches": [],
            "aggregate": None,
        }

    try:
        logs = capture_pm2_events(cfg.pm2_out_log, cfg.pm2_err_log)
    except Exception as e:
        errors.append(f"logs: {e}")
        logs = {"error": str(e), "received": [], "finished": [], "counts": {}}

    paths = save_snapshot(
        cfg.out_dir,
        scores=scores,
        logs=logs,
        meta={
            "uid": cfg.uid,
            "hotkey": cfg.hotkey,
            "netuid": cfg.netuid,
            "pm2_name": cfg.pm2_name,
            "errors": errors,
        },
    )

    agg = scores.get("aggregate") or {}
    newest = (scores.get("matches") or [{}])[0] if scores.get("matches") else {}
    summary = {
        "ok": not errors,
        "uid": cfg.uid,
        "paths": paths,
        "wandb_matches": len(scores.get("matches") or []),
        "wandb_mean": agg.get("mean"),
        "wandb_newest_final": newest.get("final_miner_score"),
        "mine_finished": (logs.get("counts") or {}).get("finished"),
        "over_budget": (logs.get("counts") or {}).get("over_budget"),
        "errors": errors,
    }
    return summary


def run_loop(cfg: MonitorConfig | None = None, *, once: bool = False) -> None:
    cfg = cfg or default_config()
    print(
        f"[monitor] uid={cfg.uid} hotkey={cfg.hotkey[:12]}… "
        f"interval={cfg.interval_s}s out={cfg.out_dir}",
        flush=True,
    )
    while True:
        started = time.perf_counter()
        try:
            summary = run_once(cfg)
            print(json.dumps({"event": "tick", **summary}, default=str), flush=True)
        except Exception:
            print(
                json.dumps(
                    {
                        "event": "tick_error",
                        "error": traceback.format_exc()[-500:],
                    }
                ),
                flush=True,
            )
        if once:
            return
        slept = max(1.0, cfg.interval_s - (time.perf_counter() - started))
        time.sleep(slept)
