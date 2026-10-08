#!/usr/bin/env python3
"""Replay a saved miner problem through CompetitiveMinerLib (offline)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass

from custom_miner.problem_store import problems_dir
from conversationgenome.task.task_factory import parse_task


async def replay(path: Path) -> int:
    data = json.loads(path.read_text())
    cgp = data.get("cgp_input")
    if not cgp:
        print("No cgp_input in file")
        return 2
    task = parse_task(cgp[0]["task"])
    print(f"type={task.type} file={path.name}")

    from custom_miner.miner_lib import CompetitiveMinerLib
    from local_dev.backends import use_miner_backend_env

    ml = CompetitiveMinerLib()
    t0 = time.perf_counter()
    with use_miner_backend_env():
        result = await ml.do_mining(task)
    elapsed = time.perf_counter() - t0
    tags = (result or {}).get("tags") if isinstance(result, dict) else None
    print(f"mine_s={elapsed:.2f}")
    if tags is not None:
        print(f"tags({len(tags)}): {', '.join(map(str, tags[:20]))}")
    else:
        print("result keys:", list((result or {}).keys()) if isinstance(result, dict) else type(result))
    out = path.with_suffix(".replay.json")
    out.write_text(json.dumps({"source": path.name, "elapsed_s": elapsed, "result": result}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="Path or filename under tasks/")
    p.add_argument("--latest", action="store_true", help="Replay most recent saved problem")
    args = p.parse_args()

    if args.latest:
        tasks = sorted((problems_dir() / "tasks").glob("*.json"))
        if not tasks:
            print("No saved tasks")
            return 1
        path = tasks[-1]
    elif args.file:
        path = Path(args.file)
        if not path.is_file():
            path = problems_dir() / "tasks" / args.file
    else:
        p.print_help()
        return 2

    return asyncio.run(replay(path))


if __name__ == "__main__":
    raise SystemExit(main())
