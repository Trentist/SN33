#!/usr/bin/env python3
"""Rebuild custom_miner/tag_pipeline/data/tag_bank.json from prepared bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREP = _ROOT / "local_dev" / "cache" / "miner_problems" / "prepared_bundles"
DEFAULT_OUT = _ROOT / "custom_miner" / "tag_pipeline" / "data" / "tag_bank.json"

EXTRAS = [
    "la county",
    "living in la",
    "southern california",
    "nyc",
    "manhattan townhouse",
    "cosby estate",
    "cosby family",
    "hardwire news",
    "cosmicbook news",
    "mixed martial arts",
    "ultimate fighting championship",
    "seattle washington",
    "city of seattle",
    "seattle council",
    "washington state",
    "comprehensive plan phase iii",
    "seattle city hall",
    "seattle urban villages",
    "warhammer 40",
    "games workshop",
    "space marines",
    "ultramarines",
    "chaos space marines",
    "adeptus custodes",
    "astra militarum",
    "balance dataslate",
    "remote work",
    "hybrid work",
    "circuit breaker",
    "fault tolerance",
    "bulkhead pattern",
    "exponential backoff",
    "distributed tracing",
    "microservice architecture",
]


def build(prep_dir: Path, out_path: Path) -> int:
    tags: set[str] = set(EXTRAS)
    if prep_dir.is_dir():
        for p in prep_dir.glob("*.json"):
            d = json.loads(p.read_text())
            meta = (d.get("input") or {}).get("metadata") or {}
            for t in meta.get("tags") or []:
                t = str(t).strip().lower()
                if 2 < len(t) < 64 and not t.isdigit():
                    tags.add(t)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source": "prepared_bundles_gt + alias_extras",
        "count": len(tags),
        "tags": sorted(tags),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} ({len(tags)} tags)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    return build(args.prep_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
