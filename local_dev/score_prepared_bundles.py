#!/usr/bin/env python3
"""Score all prepared_bundles with CompetitiveMinerLib (live embeddings)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass

# Prevent backends from reloading .env mid-run with stale values
import local_dev.backends as backends

backends._ensure_dotenv_loaded = lambda: None  # type: ignore[assignment]

from conversationgenome.ConfigLib import c
from conversationgenome.task_bundle.task_bundle_factory import try_parse_task_bundle
from local_dev.backends import make_backend
from local_dev.backends import override_get_llm_backend
from local_dev.backends import validator_llm
from local_dev.problem_view import DEFAULT_OUT_DIR as PROBLEMS_DIR
from local_dev.problem_view import refresh_index
from local_dev.problem_view import write_score_result

PREP = _ROOT / "local_dev" / "cache" / "miner_problems" / "prepared_bundles"
RUN_LABEL = os.environ.get("SCORE_RUN_LABEL") or "latest"
OUT = Path(
    os.environ.get("SCORE_OUT")
    or str(_ROOT / "local_dev" / "cache" / "miner_problems" / f"score_{RUN_LABEL}.json")
)
# SCORE_MODEL wins (not in .env) so CLI overrides survive load_dotenv(override=True)
MODEL = (
    os.environ.get("SCORE_MODEL")
    or os.environ.get("MINER_OPENROUTER_MODEL")
    or os.environ.get("OPENROUTER_MODEL")
    or "google/gemini-2.5-flash"
)


async def run_one(path: Path) -> dict:
    import custom_miner.prompt_overrides as po

    po._installed = False
    from custom_miner.miner_lib import CompetitiveMinerLib

    b = try_parse_task_bundle(json.loads(path.read_text()))
    if not b:
        return {"file": path.name, "type": "?", "score": 0.0, "error": "parse failed"}

    ml = CompetitiveMinerLib()
    os.environ["OPENROUTER_MODEL"] = MODEL
    c.set("env", "OPENROUTER_MODEL", MODEL)
    miner_llm = make_backend("openrouter")
    tasks = b.to_mining_tasks(1)
    if not tasks:
        return {"file": path.name, "type": b.type, "score": 0.0, "error": "no tasks"}

    task = tasks[0]
    t0 = time.perf_counter()
    with override_get_llm_backend(miner_llm):
        mined = await ml.do_mining(task)
    mine_s = time.perf_counter() - t0
    raw_tags = list((mined or {}).get("tags") or []) if isinstance(mined, dict) else []

    vali = validator_llm()
    formatted = None
    entry: dict = {}
    last_err: Exception | None = None
    for _attempt in range(3):
        try:
            with override_get_llm_backend(vali):
                formatted = await b.format_results(dict(mined or {}))
                resp = SimpleNamespace(
                    cgp_output=[formatted],
                    axon=SimpleNamespace(uuid="x", hotkey="custom"),
                )
                scores, _ = await b.evaluate(miner_responses=[resp])
            entry = (scores or [{}])[0]
            last_err = None
            break
        except Exception as e:
            last_err = e
            await asyncio.sleep(1.5 * (_attempt + 1))
    if last_err is not None:
        # Fallback: score raw miner tags when validator validate_tag_set flakes
        formatted = dict(mined or {})
        if isinstance(formatted, dict) and raw_tags and not formatted.get("tags"):
            formatted["tags"] = raw_tags
        with override_get_llm_backend(vali):
            resp = SimpleNamespace(
                cgp_output=[formatted],
                axon=SimpleNamespace(uuid="x", hotkey="custom"),
            )
            scores, _ = await b.evaluate(miner_responses=[resp])
        entry = (scores or [{}])[0]
        entry = dict(entry)
        entry["validate_fallback"] = str(last_err)
    return {
        "file": path.name,
        "type": getattr(b, "type", None),
        "score": float(entry.get("final_miner_score") or 0.0),
        "adjusted": float(entry.get("adjustedScore") or 0.0),
        "mine_s": mine_s,
        "raw_tags": raw_tags[:20],
        "scored_tags": list((formatted or {}).get("tags") or [])[:20]
        if isinstance(formatted, dict)
        else [],
        "section_coverage": entry.get("section_coverage"),
        "skill_coverage": entry.get("skill_coverage"),
        "n_tests": sum(len(v) for v in ((formatted or {}).get("section_tests") or {}).values())
        if isinstance(formatted, dict)
        else 0,
        "skill_chars": len((formatted or {}).get("skill") or "") if isinstance(formatted, dict) else 0,
        "gt_sample": list(getattr(getattr(b.input, "metadata", None), "tags", None) or [])[:12],
        "gt_tags": list(getattr(getattr(b.input, "metadata", None), "tags", None) or []),
        "model": MODEL,
    }


async def main() -> None:
    paths = sorted(PREP.glob("*.json"))
    limit = int(os.environ.get("SCORE_LIMIT", "0") or "0")
    if limit > 0:
        paths = paths[:limit]
    print(f"Scoring {len(paths)} prepared bundles with {MODEL}", flush=True)
    print(f"Per-problem cache → {PROBLEMS_DIR}/<stem>/{{raw,gt,miner,metrics,compare}}.json", flush=True)
    rows = []
    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {path.name}", flush=True)
        try:
            r = await run_one(path)
        except Exception as e:
            traceback.print_exc()
            r = {
                "file": path.name,
                "type": "?",
                "score": 0.0,
                "error": str(e),
                "mine_s": 0,
                "model": MODEL,
            }
        rows.append(r)
        print(
            f"  → type={r.get('type')} score={r.get('score', 0):.4f} adj={r.get('adjusted', 0):.4f} "
            f"mine={r.get('mine_s', 0):.1f}s tags={len(r.get('raw_tags') or [])} "
            f"err={r.get('error')}",
            flush=True,
        )
        if r.get("raw_tags"):
            print("    tags:", ", ".join(r["raw_tags"][:10]), flush=True)
        elif r.get("n_tests"):
            print(f"    skill tests={r.get('n_tests')} chars={r.get('skill_chars')}", flush=True)

        # Update cache/problems immediately after each problem so runs are eyeballable live
        stem = Path(r.get("file") or path.name).stem
        folder = write_score_result(
            out_dir=PROBLEMS_DIR,
            stem=stem,
            score_row=r,
            gt_tags=list(r.get("gt_tags") or r.get("gt_sample") or []),
            model=MODEL,
            run_label=RUN_LABEL,
            refresh_raw_gt=True,
            update_index=True,
        )
        print(f"  → cache {folder.relative_to(_ROOT)}/", flush=True)

    by = defaultdict(list)
    for r in rows:
        by[r.get("type") or "?"].append(r.get("score") or 0.0)
    print("\n======== SUMMARY ========")
    all_scores = [r.get("score") or 0.0 for r in rows]
    print(
        f"n={len(rows)} overall_mean={sum(all_scores) / len(all_scores):.4f} "
        f"min={min(all_scores):.4f} max={max(all_scores):.4f}"
    )
    for t, xs in sorted(by.items()):
        print(f"  {t:32} n={len(xs)} mean={sum(xs) / len(xs):.4f} min={min(xs):.4f} max={max(xs):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "run_label": RUN_LABEL, "results": rows}, indent=2))
    summary = refresh_index(out_dir=PROBLEMS_DIR, run_label=RUN_LABEL)
    print("wrote", OUT)
    print(
        f"per-problem view → {summary['out_dir']}/<stem>/ "
        f"(mean={summary.get('mean_score')})"
    )


if __name__ == "__main__":
    asyncio.run(main())
