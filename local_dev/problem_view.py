#!/usr/bin/env python3
"""Per-problem cache layout for easy eyeballing.

Each problem lives under local_dev/cache/problems/<stem>/:

  raw.json      original reserved ReadyAI bundle
  gt.json       ground-truth tags / metadata (from prepared bundle)
  miner.json    our miner tags / output (latest score run)
  metrics.json  score + overlap / miss analysis vs GT

Usage:
  python -m local_dev.problem_view              # rebuild from existing caches
  python -m local_dev.problem_view --score PATH # also seed miner+metrics from a score dump
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = _ROOT / "local_dev" / "cache" / "testnet_problems"
DEFAULT_PREP_DIR = _ROOT / "local_dev" / "cache" / "miner_problems" / "prepared_bundles"
DEFAULT_OUT_DIR = _ROOT / "local_dev" / "cache" / "problems"


def _stem(name: str) -> str:
    return Path(name).stem


def _norm(tag: str) -> str:
    t = (tag or "").strip().lower()
    t = t.replace(".", " ").replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def extract_gt(prepared: dict[str, Any]) -> dict[str, Any]:
    inp = prepared.get("input") or {}
    meta = inp.get("metadata") or {}
    tags = list(meta.get("tags") or [])
    out: dict[str, Any] = {
        "guid": prepared.get("guid") or inp.get("guid"),
        "type": prepared.get("type"),
        "scoring_mechanism": prepared.get("scoring_mechanism"),
        "tags": tags,
        "n_tags": len(tags),
        "example_output": prepared.get("example_output"),
    }
    # Skill / section extras when present
    data = inp.get("data") or {}
    if isinstance(data, dict) and data.get("section_map"):
        out["section_ids"] = [
            s.get("section_id") if isinstance(s, dict) else str(s) for s in data["section_map"]
        ]
    if meta.get("participantProfiles"):
        out["participant_profiles"] = meta["participantProfiles"]
    return out


def analyze_tags(
    *,
    gt_tags: list[str],
    miner_tags: list[str],
    scored_tags: list[str] | None = None,
) -> dict[str, Any]:
    gt_norm = {_norm(t): t for t in gt_tags if t}
    miner_norm = {_norm(t): t for t in miner_tags if t}
    scored = scored_tags if scored_tags is not None else miner_tags
    scored_norm = {_norm(t): t for t in scored if t}

    exact_hits = sorted(gt_norm[k] for k in gt_norm.keys() & miner_norm.keys())
    missed_gt = sorted(gt_norm[k] for k in gt_norm.keys() - miner_norm.keys())
    unique_miner = sorted(miner_norm[k] for k in miner_norm.keys() - gt_norm.keys())
    scored_hits = sorted(gt_norm[k] for k in gt_norm.keys() & scored_norm.keys())

    n_gt = len(gt_norm)
    n_miner = len(miner_norm)
    return {
        "n_gt": n_gt,
        "n_miner": n_miner,
        "n_scored": len(scored_norm),
        "exact_hit_count": len(exact_hits),
        "exact_hits": exact_hits,
        "missed_gt": missed_gt,
        "unique_miner": unique_miner,
        "scored_exact_hits": scored_hits,
        "gt_coverage": (len(exact_hits) / n_gt) if n_gt else None,
        "miner_precision_exact": (len(exact_hits) / n_miner) if n_miner else None,
    }


def write_problem_folder(
    *,
    out_dir: Path,
    stem: str,
    raw: dict[str, Any] | None = None,
    gt: dict[str, Any] | None = None,
    miner: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    folder = out_dir / stem
    folder.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        _write_json(folder / "raw.json", raw)
    if gt is not None:
        _write_json(folder / "gt.json", gt)
    if miner is not None:
        _write_json(folder / "miner.json", miner)
    if metrics is not None:
        _write_json(folder / "metrics.json", metrics)
    return folder


def write_score_result(
    *,
    out_dir: Path,
    stem: str,
    score_row: dict[str, Any],
    gt_tags: list[str] | None = None,
    model: str | None = None,
    run_label: str | None = None,
    refresh_raw_gt: bool = True,
    raw_dir: Path = DEFAULT_RAW_DIR,
    prep_dir: Path = DEFAULT_PREP_DIR,
    update_index: bool = False,
) -> Path:
    """Update per-problem cache after a score run.

    Always writes miner.json + metrics.json. When refresh_raw_gt=True (default),
    also refreshes raw.json + gt.json from testnet/prepared caches so every test
    run leaves a complete eyeball folder.
    """
    folder = out_dir / stem
    folder.mkdir(parents=True, exist_ok=True)

    raw = None
    gt = None
    if refresh_raw_gt:
        raw_path = raw_dir / f"{stem}.json"
        prep_path = prep_dir / f"{stem}.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
        if prep_path.exists():
            gt = extract_gt(json.loads(prep_path.read_text()))

    if gt_tags is None:
        if gt and gt.get("tags"):
            gt_tags = list(gt["tags"])
        else:
            gt_path = folder / "gt.json"
            if gt_path.exists():
                gt_tags = list((json.loads(gt_path.read_text()) or {}).get("tags") or [])
            else:
                gt_tags = list(score_row.get("gt_tags") or score_row.get("gt_sample") or [])

    raw_tags = list(score_row.get("raw_tags") or [])
    scored_tags = list(score_row.get("scored_tags") or [])
    overlap = analyze_tags(gt_tags=gt_tags, miner_tags=raw_tags, scored_tags=scored_tags or None)
    now = datetime.now(timezone.utc).isoformat()

    miner = {
        "run_label": run_label,
        "model": model or score_row.get("model"),
        "type": score_row.get("type"),
        "raw_tags": raw_tags,
        "scored_tags": scored_tags,
        "section_coverage": score_row.get("section_coverage"),
        "skill_coverage": score_row.get("skill_coverage"),
        "n_tests": score_row.get("n_tests"),
        "skill_chars": score_row.get("skill_chars"),
        "error": score_row.get("error"),
        "updated_at": now,
    }
    metrics = {
        "run_label": run_label,
        "model": miner["model"],
        "type": score_row.get("type"),
        "score": score_row.get("score"),
        "adjusted": score_row.get("adjusted"),
        "mine_s": score_row.get("mine_s"),
        "error": score_row.get("error"),
        **overlap,
        "updated_at": now,
    }
    # Side-by-side for quick eyeballing
    compare = {
        "run_label": run_label,
        "model": miner["model"],
        "score": score_row.get("score"),
        "gt_tags": list(gt_tags or []),
        "miner_tags": raw_tags,
        "scored_tags": scored_tags,
        "exact_hits": overlap["exact_hits"],
        "missed_gt": overlap["missed_gt"],
        "unique_miner": overlap["unique_miner"],
        "updated_at": now,
    }

    write_problem_folder(out_dir=out_dir, stem=stem, raw=raw, gt=gt, miner=miner, metrics=metrics)
    _write_json(folder / "compare.json", compare)

    if update_index:
        refresh_index(out_dir=out_dir, run_label=run_label)
    return folder


def refresh_index(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Rewrite problems/index.json from current per-problem folders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for folder in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        stem = folder.name
        gt = json.loads((folder / "gt.json").read_text()) if (folder / "gt.json").exists() else {}
        metrics = (
            json.loads((folder / "metrics.json").read_text())
            if (folder / "metrics.json").exists()
            else {}
        )
        index.append(
            {
                "stem": stem,
                "type": gt.get("type") or metrics.get("type"),
                "guid": gt.get("guid"),
                "score": metrics.get("score"),
                "exact_hit_count": metrics.get("exact_hit_count"),
                "n_gt": metrics.get("n_gt") or gt.get("n_tags"),
                "n_miner": metrics.get("n_miner"),
                "missed_gt": metrics.get("missed_gt"),
                "folder": str(folder.relative_to(_ROOT)),
            }
        )
    scores = [x["score"] for x in index if isinstance(x.get("score"), (int, float))]
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir.relative_to(_ROOT)),
        "n_problems": len(index),
        "n_scored": len([s for s in scores if s is not None]),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "run_label": run_label,
        "layout": {
            "raw.json": "original ReadyAI bundle",
            "gt.json": "ground-truth tags / metadata",
            "miner.json": "our miner tags (latest run)",
            "metrics.json": "score + exact overlap vs GT",
            "compare.json": "side-by-side gt vs miner hits/misses",
        },
        "problems": index,
    }
    _write_json(out_dir / "index.json", summary)
    return summary


def rebuild(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    prep_dir: Path = DEFAULT_PREP_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    score_path: Path | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_by_file: dict[str, dict[str, Any]] = {}
    model = None
    if score_path and score_path.exists():
        rows = json.loads(score_path.read_text())
        if isinstance(rows, dict) and "results" in rows:
            model = rows.get("model")
            rows = rows["results"]
        for r in rows or []:
            if isinstance(r, dict) and r.get("file"):
                score_by_file[r["file"]] = r
        run_label = run_label or score_path.stem

    index: list[dict[str, Any]] = []
    prep_files = sorted(prep_dir.glob("*.json")) if prep_dir.exists() else []
    # Fall back to raw dir stems if no prepared bundles
    stems = [p.stem for p in prep_files] or [
        p.stem for p in sorted(raw_dir.glob("*.json")) if p.name != "index.json"
    ]

    for stem in stems:
        raw_path = raw_dir / f"{stem}.json"
        prep_path = prep_dir / f"{stem}.json"
        raw = json.loads(raw_path.read_text()) if raw_path.exists() else None
        prepared = json.loads(prep_path.read_text()) if prep_path.exists() else None
        gt = extract_gt(prepared) if prepared else None

        score_row = score_by_file.get(f"{stem}.json")
        if score_row:
            write_score_result(
                out_dir=out_dir,
                stem=stem,
                score_row=score_row,
                gt_tags=list((gt or {}).get("tags") or []),
                model=model or score_row.get("model"),
                run_label=run_label,
            )
        else:
            # Placeholders so every folder always has the same four files
            write_problem_folder(
                out_dir=out_dir,
                stem=stem,
                miner={"raw_tags": [], "scored_tags": [], "note": "not scored yet"},
                metrics={"score": None, "note": "not scored yet"},
            )

        # Always refresh raw + gt
        write_problem_folder(out_dir=out_dir, stem=stem, raw=raw, gt=gt)

        metrics_path = out_dir / stem / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        index.append(
            {
                "stem": stem,
                "type": (gt or {}).get("type") or (score_row or {}).get("type"),
                "guid": (gt or {}).get("guid"),
                "score": (metrics or {}).get("score"),
                "exact_hit_count": (metrics or {}).get("exact_hit_count"),
                "n_gt": (metrics or {}).get("n_gt"),
                "n_miner": (metrics or {}).get("n_miner"),
                "folder": str((out_dir / stem).relative_to(_ROOT)),
            }
        )

    # Problems that had no score still need index entries
    for stem in stems:
        if any(x["stem"] == stem for x in index):
            continue
        gt_path = out_dir / stem / "gt.json"
        gt = json.loads(gt_path.read_text()) if gt_path.exists() else {}
        index.append(
            {
                "stem": stem,
                "type": gt.get("type"),
                "guid": gt.get("guid"),
                "score": None,
                "exact_hit_count": None,
                "n_gt": gt.get("n_tags"),
                "n_miner": None,
                "folder": str((out_dir / stem).relative_to(_ROOT)),
            }
        )

    return refresh_index(out_dir=out_dir, run_label=run_label)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild per-problem cache view")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--score",
        type=Path,
        default=None,
        help="Optional score_*.json to seed miner.json + metrics.json",
    )
    ap.add_argument("--run-label", type=str, default=None)
    args = ap.parse_args()

    # Prefer latest mercury sample if --score omitted and files exist
    score = args.score
    if score is None:
        candidates = [
            DEFAULT_PREP_DIR.parent / "score_mercury2_sample.json",
            DEFAULT_PREP_DIR.parent / "score_mercury2_high.json",
            DEFAULT_PREP_DIR.parent / "score_robust_v5.json",
        ]
        for c in candidates:
            if c.exists():
                score = c
                break

    summary = rebuild(
        raw_dir=args.raw_dir,
        prep_dir=args.prep_dir,
        out_dir=args.out_dir,
        score_path=score,
        run_label=args.run_label,
    )
    print(
        f"wrote {summary['n_problems']} problems → {summary['out_dir']} "
        f"(scored={summary['n_scored']} mean={summary['mean_score']})"
    )
    print("each folder: raw.json  gt.json  miner.json  metrics.json")


if __name__ == "__main__":
    main()
