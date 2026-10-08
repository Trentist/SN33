#!/usr/bin/env python3
"""Build miner-facing problems from cached ReadyAI bundles.

Takes reserved raw bundles, runs the same validator setup() path (quality/GT/
section map), then emits the masked Task payloads that validators put in
CgSynapse.cgp_input — i.e. exactly what miners receive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass

from local_dev.api_source import DEFAULT_CACHE_DIR
from local_dev.api_source import list_cached_problems
from local_dev.backends import override_get_llm_backend
from local_dev.backends import validator_llm

DEFAULT_MINER_DIR = _ROOT / "local_dev" / "cache" / "miner_problems"


def _task_to_dict(task) -> dict[str, Any]:
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json")
    return dict(task)


def _bundle_meta_summary(bundle) -> dict[str, Any]:
    meta = getattr(bundle.input, "metadata", None)
    out: dict[str, Any] = {
        "bundle_guid": str(getattr(bundle, "guid", None)),
        "type": bundle.type,
        "quality_score": getattr(bundle.input, "quality_score", None),
        "ready": bool(bundle.is_ready()),
    }
    if meta is None:
        out["gt_tags"] = []
        return out
    tags = getattr(meta, "tags", None)
    if tags:
        out["gt_tags"] = list(tags)
    sections = getattr(bundle.input.data, "section_map", None) or getattr(meta, "sections", None)
    if sections:
        out["section_ids"] = [getattr(s, "section_id", str(s)) for s in sections]
    return out


async def build_miner_problems(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    out_dir: Path = DEFAULT_MINER_DIR,
    tasks_per_bundle: int = 2,
    limit: int | None = None,
) -> Path:
    from conversationgenome.task_bundle.task_bundle_factory import try_parse_task_bundle

    problems = list_cached_problems(cache_dir)
    if not problems:
        raise FileNotFoundError(f"No cached problems in {cache_dir}. Run: python -m local_dev.fetch_problems --count 20")

    if limit is not None:
        problems = problems[:limit]

    miner_dir = out_dir / "tasks"
    prepared_dir = out_dir / "prepared_bundles"
    miner_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    vali = validator_llm()
    index: list[dict[str, Any]] = []
    task_count = 0

    with override_get_llm_backend(vali):
        for i, row in enumerate(problems):
            raw_path = cache_dir / row["file"]
            print(f"\n[{i+1}/{len(problems)}] Preparing {row['type']} guid={row['guid']}")
            raw = json.loads(raw_path.read_text())
            bundle = try_parse_task_bundle(raw)
            if not bundle:
                print("  SKIP: parse failed")
                continue

            try:
                await bundle.setup()
            except Exception as e:
                print(f"  SKIP: setup failed: {e}")
                continue

            summary = _bundle_meta_summary(bundle)
            if not bundle.is_ready():
                print(f"  SKIP: not ready after setup (quality={summary.get('quality_score')})")
                index.append({**summary, "status": "not_ready", "miner_tasks": []})
                continue

            # Persist prepared bundle (includes GT metadata) for later scoring
            try:
                prepared_path = prepared_dir / f"{i:02d}_{bundle.type}_{bundle.guid}.json"
                prepared_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2))
            except Exception as e:
                print(f"  WARN: could not serialize prepared bundle: {e}")
                prepared_path = None

            mining_tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=tasks_per_bundle)
            miner_task_files = []
            for j, task in enumerate(mining_tasks):
                masked = bundle.mask_task_for_miner(task)
                # Exact shape validators put into CgSynapse.cgp_input
                synapse_input = {"task": _task_to_dict(masked)}
                fname = f"{i:02d}_{j:02d}_{bundle.type}_{bundle.guid}.json"
                path = miner_dir / fname
                path.write_text(
                    json.dumps(
                        {
                            "cgp_input": [synapse_input],
                            "bundle_guid": str(bundle.guid),
                            "bundle_type": bundle.type,
                            "task_index": j,
                            "source_file": row["file"],
                            "prepared_bundle_file": prepared_path.name if prepared_path else None,
                        },
                        indent=2,
                    )
                )
                miner_task_files.append(fname)
                task_count += 1
                print(f"  + miner task {j}: {fname} type={masked.type}")

            index.append(
                {
                    **summary,
                    "status": "ok",
                    "source_file": row["file"],
                    "prepared_bundle_file": prepared_path.name if prepared_path else None,
                    "miner_tasks": miner_task_files,
                }
            )

    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "tasks_per_bundle": tasks_per_bundle,
                "bundles": index,
                "total_miner_tasks": task_count,
                "tasks_dir": str(miner_dir),
                "prepared_dir": str(prepared_dir),
            },
            indent=2,
        )
    )
    print(f"\nDone. {task_count} miner tasks from {len(index)} bundles → {out_dir}")
    return out_dir


def list_miner_problems(out_dir: Path = DEFAULT_MINER_DIR) -> None:
    index_path = out_dir / "index.json"
    if not index_path.is_file():
        print(f"No miner problems at {out_dir}. Run: python -m local_dev.build_miner_problems")
        return
    data = json.loads(index_path.read_text())
    print(f"total_miner_tasks={data.get('total_miner_tasks')}  tasks_per_bundle={data.get('tasks_per_bundle')}")
    for b in data.get("bundles") or []:
        print(
            f"  [{b.get('status'):9}] {b.get('type'):30} guid={b.get('bundle_guid')} "
            f"tasks={len(b.get('miner_tasks') or [])} quality={b.get('quality_score')}"
        )


async def _amain() -> int:
    p = argparse.ArgumentParser(description="Build miner-facing tasks from cached ReadyAI problems")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MINER_DIR)
    p.add_argument("--tasks-per-bundle", type=int, default=2, help="Windows/tasks per bundle (live validators use ~5)")
    p.add_argument("--limit", type=int, default=None, help="Only process first N cached bundles")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        list_miner_problems(args.out_dir)
        return 0

    await build_miner_problems(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        tasks_per_bundle=args.tasks_per_bundle,
        limit=args.limit,
    )
    list_miner_problems(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
