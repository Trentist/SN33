#!/usr/bin/env python3
"""CLI: python -m local_dev.monitor [--once] [--interval 120]"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass


def main() -> int:
    from local_dev.monitor.config import default_config
    from local_dev.monitor.runner import run_loop
    from local_dev.monitor.runner import run_once

    p = argparse.ArgumentParser(description="Auto-fetch WandB scores + pm2 miner logs for SN33 UID")
    p.add_argument("--once", action="store_true", help="Single poll then exit")
    p.add_argument("--interval", type=float, default=None, help="Seconds between polls (default 120)")
    p.add_argument("--uid", type=int, default=None)
    p.add_argument("--hotkey", type=str, default=None)
    p.add_argument("--pm2-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--json", action="store_true", help="With --once, print full summary JSON")
    args = p.parse_args()

    if args.uid is not None:
        os.environ["MONITOR_UID"] = str(args.uid)
    if args.hotkey:
        os.environ["MONITOR_HOTKEY"] = args.hotkey
    if args.pm2_name:
        os.environ["MONITOR_PM2_NAME"] = args.pm2_name
    if args.out_dir:
        os.environ["MONITOR_OUT_DIR"] = args.out_dir
    if args.interval is not None:
        os.environ["MONITOR_INTERVAL_S"] = str(args.interval)

    cfg = default_config()
    if args.once:
        summary = run_once(cfg)
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(
                f"uid={summary['uid']} matches={summary['wandb_matches']} "
                f"mean={summary.get('wandb_mean')} newest={summary.get('wandb_newest_final')} "
                f"finished={summary.get('mine_finished')} over={summary.get('over_budget')} "
                f"→ {summary['paths']['latest']}"
            )
            if summary.get("errors"):
                print("errors:", summary["errors"], file=sys.stderr)
        return 0 if summary.get("ok") else 1

    run_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
