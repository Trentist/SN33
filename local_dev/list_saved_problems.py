#!/usr/bin/env python3
"""List / inspect miner problems saved from live axon requests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_miner.problem_store import list_saved_problems
from custom_miner.problem_store import problems_dir


def main() -> int:
    p = argparse.ArgumentParser(description="List saved miner request problems")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--show", type=str, default=None, help="Filename under tasks/ to print")
    p.add_argument("--stats", action="store_true")
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Override problems dir (sets MINER_PROBLEMS_DIR for this process)",
    )
    p.add_argument("--mainnet", action="store_true", help="Use local_dev/cache/mainnet_problems")
    p.add_argument("--testnet", action="store_true", help="Use local_dev/cache/testnet_live_problems")
    args = p.parse_args()

    if args.dir:
        import os

        os.environ["MINER_PROBLEMS_DIR"] = str(args.dir.resolve())
    elif args.mainnet:
        import os

        os.environ["MINER_PROBLEMS_DIR"] = str(_ROOT / "local_dev" / "cache" / "mainnet_problems")
    elif args.testnet:
        import os

        os.environ["MINER_PROBLEMS_DIR"] = str(_ROOT / "local_dev" / "cache" / "testnet_live_problems")

    out = problems_dir()
    print(f"problems_dir={out}")
    index = out / "index.json"
    if not index.is_file():
        print("No saved problems yet. Run miner with SAVE_MINER_PROBLEMS=1 on mainnet/testnet.")
        return 0

    if args.show:
        path = out / "tasks" / args.show
        if not path.is_file():
            # allow bare path
            path = Path(args.show)
        print(path.read_text()[:8000])
        return 0

    rows = list_saved_problems(limit=args.limit if not args.stats else 5000)
    if args.stats:
        types = Counter(r.get("task_type") for r in rows)
        nets = Counter(str(r.get("netuid")) for r in rows)
        print(f"count={len(rows)}")
        print("by_type:", dict(types))
        print("by_netuid:", dict(nets))
        ok = sum(1 for r in rows if r.get("has_result"))
        print(f"with_result={ok}")
        return 0

    for r in rows:
        print(
            f"{r.get('saved_at','?'):25} type={r.get('task_type'):30} "
            f"netuid={r.get('netuid')} file={r.get('file')} "
            f"result={'Y' if r.get('has_result') else 'N'} "
            f"t={r.get('elapsed_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
