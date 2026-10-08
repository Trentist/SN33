#!/usr/bin/env python3
"""Fetch N live testnet problems from ReadyAI and cache them for local testing."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass


async def _main() -> int:
    from local_dev.api_source import DEFAULT_CACHE_DIR
    from local_dev.api_source import fetch_and_cache_problems
    from local_dev.api_source import list_cached_problems

    p = argparse.ArgumentParser(description="Cache live ReadyAI testnet problems")
    p.add_argument("--count", type=int, default=20, help="How many problems to reserve (default 20)")
    p.add_argument("--netuid", type=int, default=138)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--task",
        default=None,
        help="Keep reserving until this task type is cached (e.g. webpage_metadata_generation)",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Reserve attempts when --task is set (default count * 15)",
    )
    p.add_argument("--list", action="store_true", help="List already cached problems and exit")
    args = p.parse_args()

    if args.list:
        problems = list_cached_problems(args.cache_dir)
        if not problems:
            print(f"No cached problems in {args.cache_dir}")
            return 1
        for i, row in enumerate(problems):
            print(f"{i:02d}  {row.get('type'):30}  guid={row.get('guid')}  {row.get('file')}")
        print(f"Total: {len(problems)} in {args.cache_dir}")
        return 0

    await fetch_and_cache_problems(
        count=args.count,
        netuid=args.netuid,
        cache_dir=args.cache_dir,
        task_type=args.task,
        max_attempts=args.max_attempts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
