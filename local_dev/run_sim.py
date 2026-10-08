#!/usr/bin/env python3
"""CLI for local SN33 miner development (no Bittensor chain).

Default: reserve a LIVE testnet problem from ReadyAI API, run validator-matched
GT/quality/embed models, mine with Gemini 2.5 Flash, score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local SN33 miner development harness")
    p.add_argument(
        "--mode",
        choices=["mock", "live", "validator"],
        default="validator",
        help="validator=full live-matched GT→mine→score (default)",
    )
    p.add_argument("--miner", choices=["baseline", "custom"], default="custom")
    p.add_argument(
        "--source",
        choices=["api", "fixture", "cache"],
        default="api",
        help="api=live reserve; cache=local_dev/cache; fixture=DummyData",
    )
    p.add_argument("--cache-index", type=int, default=0, help="Which cached problem to run (with --source cache)")
    p.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="ReadyAI netuid (default from READYAI_NETUID or 138)",
    )
    p.add_argument(
        "--task",
        default="any",
        choices=[
            "any",
            "conversation_tagging",
            "webpage_metadata_generation",
            "survey_tagging",
            "named_entities_extraction",
            "skill_generation",
            "skill_coverage_evaluation",
        ],
        help="Filter API reserves to this type (any = first bundle)",
    )
    p.add_argument("--skip-quality", action="store_true", help="(legacy) ignored on API setup path")
    p.add_argument("--compare", action="store_true", help="(mock/live) baseline vs custom")
    p.add_argument("--windows", type=int, default=1, help="Mining windows / tasks per bundle")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    p.add_argument("--models", action="store_true", help="Print configured model map and exit")
    return p.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    if args.models:
        from local_dev.backends import describe_models

        print(json.dumps(describe_models(), indent=2))
        return 0

    if args.mode == "validator":
        from local_dev.validator_loop import main_async

        return await main_async(
            windows=args.windows,
            miner_kind=args.miner,
            as_json=args.json,
            task=args.task,
            source=args.source,
            netuid=args.netuid,
            skip_quality=args.skip_quality,
            cache_index=args.cache_index,
        )

    from local_dev.harness import compare_miners
    from local_dev.harness import compare_skill_miners
    from local_dev.harness import print_results
    from local_dev.harness import run_conversation_sim
    from local_dev.harness import run_skill_coverage_sim

    if args.mode == "live" and not (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("CHUTES_API_KEY")
    ):
        print("ERROR: --mode live needs an LLM API key in .env", file=sys.stderr)
        return 2

    skill_task = args.task == "skill_coverage_evaluation"

    if args.compare:
        data = await (compare_skill_miners(mode=args.mode) if skill_task else compare_miners(mode=args.mode))
        if args.json:
            print(json.dumps({k: [x.to_dict() for x in v] for k, v in data.items()}, indent=2))
        else:
            print("=== baseline ===")
            print_results(data["baseline"])
            print("=== custom ===")
            print_results(data["custom"])
        return 0

    if skill_task:
        results = await run_skill_coverage_sim(miner=args.miner, mode=args.mode, window_limit=args.windows)
    else:
        results = await run_conversation_sim(miner=args.miner, mode=args.mode, window_limit=args.windows)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_results(results)
    return 0


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
