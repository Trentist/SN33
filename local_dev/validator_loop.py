"""Validator-like local cycle matching live SN33 validator behavior.

Sources:
  - api (default): reserve live bundles from ReadyAI testnet API (netuid 138)
  - fixture: DummyData offline samples

Conversation / tagging path (live-matched):
  1) bundle.setup() → quality gate + GT (gpt-5-mini / gpt-5.2) + windows
  2) Miner tags windows (Gemini 2.5 Flash by default)
  3) format_results + GroundTruthTagSimilarityScoringMechanism

Skill coverage path:
  1) setup() → section map via gpt-5.6-luna
  2) Miner skill bundle → format_results (judge) → SkillCoverageScoringMechanism
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Literal
from typing import Optional

from conversationgenome.task_bundle.TaskBundle import TaskBundle
from local_dev.api_source import load_cached_bundle
from local_dev.api_source import reserve_live_bundle
from local_dev.backends import describe_models
from local_dev.backends import override_get_llm_backend
from local_dev.backends import use_miner_backend_env
from local_dev.backends import validator_llm
from tests.mocks.DummyData import DummyData

SourceKind = Literal["api", "fixture", "cache"]
TaskKind = Literal[
    "conversation_tagging",
    "webpage_metadata_generation",
    "survey_tagging",
    "named_entities_extraction",
    "skill_generation",
    "skill_coverage_evaluation",
    "any",
]


@dataclass
class CycleResult:
    task_type: str
    window_idx: int
    gt_tags: list[str]
    miner_tags: list[str]
    adjusted_score: float
    final_miner_score: float
    gt_elapsed_s: float
    mine_elapsed_s: float
    validator_model: str
    miner_model: str
    quality_score: Optional[int] = None
    quality_passed: Optional[bool] = None
    source: str = "fixture"
    bundle_guid: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _mine(task, miner_kind: str = "custom") -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with use_miner_backend_env():
        if miner_kind == "custom":
            from custom_miner.miner_lib import CompetitiveMinerLib

            ml = CompetitiveMinerLib()
        else:
            from conversationgenome.miner.MinerLib import MinerLib

            ml = MinerLib()
        result = await ml.do_mining(task)
    return result or {}, time.perf_counter() - started


def _gt_summary(bundle: TaskBundle) -> list[str]:
    meta = getattr(bundle.input, "metadata", None)
    if meta is None:
        return []
    tags = getattr(meta, "tags", None)
    if tags:
        return list(tags)
    sections = getattr(meta, "sections", None)
    if sections:
        return [getattr(s, "section_id", str(s)) for s in sections]
    section_map = getattr(bundle.input.data, "section_map", None)
    if section_map:
        return [getattr(s, "section_id", str(s)) for s in section_map]
    return []


def _validator_model_for(task_type: str) -> str:
    models = describe_models()["local_dev_validator"]
    if task_type == "skill_coverage_evaluation":
        return models["skill_model"]
    return models["gt_model"]


async def _score_bundle_tasks(
    bundle: TaskBundle,
    *,
    windows: int,
    miner_kind: str,
    source: str,
    setup_elapsed: float,
) -> list[CycleResult]:
    models = describe_models()
    tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=windows)
    if not tasks:
        raise RuntimeError(f"Bundle {bundle.type} produced no mining tasks")

    gt_tags = _gt_summary(bundle)
    quality_score = getattr(bundle.input, "quality_score", None)
    out: list[CycleResult] = []

    for task in tasks:
        mined, mine_elapsed = await _mine(task, miner_kind=miner_kind)
        formatted = await bundle.format_results(dict(mined))
        response = SimpleNamespace(
            cgp_output=[formatted],
            axon=SimpleNamespace(uuid="local-miner", hotkey="local-miner"),
        )
        scores, _ = await bundle.evaluate(miner_responses=[response])
        entry = (scores or [{}])[0]

        if isinstance(formatted, dict) and "tags" in formatted:
            miner_out = list(formatted.get("tags") or [])
        elif isinstance(formatted, dict) and "section_tests" in formatted:
            miner_out = list((formatted.get("section_tests") or {}).keys())
        else:
            miner_out = []

        window_idx = getattr(getattr(task.input, "data", None), "window_idx", 0)
        out.append(
            CycleResult(
                task_type=bundle.type,
                window_idx=window_idx if window_idx is not None else 0,
                gt_tags=gt_tags,
                miner_tags=miner_out,
                adjusted_score=float(entry.get("adjustedScore") or 0.0),
                final_miner_score=float(entry.get("final_miner_score") or 0.0),
                gt_elapsed_s=setup_elapsed,
                mine_elapsed_s=mine_elapsed,
                validator_model=_validator_model_for(bundle.type),
                miner_model=models["local_dev_miner"]["model"],
                quality_score=int(quality_score) if quality_score is not None else None,
                quality_passed=True if quality_score is None else None,
                source=source,
                bundle_guid=str(getattr(bundle, "guid", None)),
                extra={
                    k: entry.get(k)
                    for k in ("section_coverage", "skill_coverage")
                    if entry.get(k) is not None
                },
            )
        )
    return out


async def _fixture_bundle(task: TaskKind) -> TaskBundle:
    if task == "skill_coverage_evaluation":
        bundle = DummyData.skill_coverage_evaluation_task_bundle()
        raw = json.loads(bundle.input.data.lines[0][1])
        bundle.input.data.seed = raw.get("seed")
        bundle.input.metadata = None
        bundle.input.data.section_map = None
        return bundle
    if task in ("any", "conversation_tagging"):
        bundle = DummyData.conversation_tagging_task_bundle()
        bundle.input.data.indexed_windows = DummyData.windows()
        bundle.input.metadata = None
        bundle.input.quality_score = None
        return bundle
    raise ValueError(f"Fixture source has no sample for task={task}; use --source api")


async def run_cycle(
    *,
    source: SourceKind = "api",
    task: TaskKind = "any",
    windows: int = 1,
    miner_kind: str = "custom",
    netuid: int | None = None,
    max_reserve_attempts: int = 8,
    cache_index: int = 0,
) -> list[CycleResult]:
    vali = validator_llm()
    netuid = int(netuid if netuid is not None else os.environ.get("READYAI_NETUID", "138"))

    bundle: TaskBundle | None = None
    if source == "fixture":
        bundle = await _fixture_bundle(task)
        print(f"Using fixture bundle type={bundle.type}")
    elif source == "cache":
        bundle = load_cached_bundle(index=cache_index)
        print(f"Using cached bundle[{cache_index}] type={bundle.type} guid={bundle.guid}")
        if task != "any" and bundle.type != task:
            raise RuntimeError(f"Cached bundle type={bundle.type} does not match wanted task={task}")
    else:
        for attempt in range(1, max_reserve_attempts + 1):
            candidate = await reserve_live_bundle(netuid=netuid)
            print(
                f"Reserved live bundle attempt {attempt}/{max_reserve_attempts}: "
                f"type={candidate.type} guid={candidate.guid} netuid={netuid}"
            )
            if task == "any" or candidate.type == task:
                bundle = candidate
                break
            print(f"  skipping (wanted {task})")
        if bundle is None:
            raise RuntimeError(
                f"Could not reserve a live bundle of type={task} after {max_reserve_attempts} attempts"
            )

    setup_started = time.perf_counter()
    with override_get_llm_backend(vali):
        # Live validator path: setup does quality gate + GT / section map
        await bundle.setup()
        setup_elapsed = time.perf_counter() - setup_started

        if not bundle.is_ready():
            q = getattr(bundle.input, "quality_score", None)
            return [
                CycleResult(
                    task_type=bundle.type,
                    window_idx=-1,
                    gt_tags=[],
                    miner_tags=[],
                    adjusted_score=0.0,
                    final_miner_score=0.0,
                    gt_elapsed_s=setup_elapsed,
                    mine_elapsed_s=0.0,
                    validator_model=_validator_model_for(bundle.type),
                    miner_model=describe_models()["local_dev_miner"]["model"],
                    quality_score=int(q) if q is not None else None,
                    quality_passed=False,
                    source=source,
                    bundle_guid=str(getattr(bundle, "guid", None)),
                    extra={"note": "bundle.setup() left bundle not ready (e.g. quality gate fail)"},
                )
            ]

        return await _score_bundle_tasks(
            bundle,
            windows=windows,
            miner_kind=miner_kind,
            source=source,
            setup_elapsed=setup_elapsed,
        )


# Back-compat aliases
async def run_validator_like_cycle(**kwargs) -> list[CycleResult]:
    return await run_cycle(**kwargs)


def print_cycle(results: list[CycleResult]) -> None:
    if not results:
        print("No results")
        return
    r0 = results[0]
    print(f"Source: {r0.source}  bundle={r0.bundle_guid}")
    print(f"Task: {r0.task_type}")
    v = describe_models()["local_dev_validator"]
    print(
        f"Validator models: GT/chat={v['gt_model']}, quality={v['quality_model']}, "
        f"skill={v['skill_model']}, embed={v['embeddings']}"
    )
    print(f"Miner model: {r0.miner_model}")
    if r0.quality_score is not None:
        print(f"Quality gate: score={r0.quality_score} passed={r0.quality_passed}")
    if r0.gt_tags:
        print(f"GT ({len(r0.gt_tags)}): {', '.join(map(str, r0.gt_tags[:20]))}")
    for r in results:
        print(
            f"[idx {r.window_idx}] score={r.final_miner_score:.4f} "
            f"adj={r.adjusted_score:.4f} mine_time={r.mine_elapsed_s:.1f}s gt_time={r.gt_elapsed_s:.1f}s"
        )
        if r.miner_tags:
            print(f"  miner out: {', '.join(map(str, r.miner_tags[:15]))}")
        if r.extra:
            print(f"  extra: {r.extra}")


async def main_async(
    windows: int = 1,
    miner_kind: str = "custom",
    as_json: bool = False,
    task: TaskKind = "any",
    source: SourceKind = "api",
    netuid: int | None = None,
    skip_quality: bool = False,  # kept for CLI compat; API path uses bundle.setup()
    cache_index: int = 0,
) -> int:
    if skip_quality:
        print("NOTE: --skip-quality is ignored for API/setup path (live setup always runs quality when applicable)")
    results = await run_cycle(
        source=source,
        task=task,
        windows=windows,
        miner_kind=miner_kind,
        netuid=netuid,
        cache_index=cache_index,
    )
    if as_json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_cycle(results)
    return 0


if __name__ == "__main__":
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Live-matched validator-like local cycle")
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--miner", choices=["custom", "baseline"], default="custom")
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
    )
    p.add_argument("--source", choices=["api", "fixture", "cache"], default="api")
    p.add_argument("--cache-index", type=int, default=0)
    p.add_argument("--netuid", type=int, default=None, help="ReadyAI netuid (default 138 testnet)")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    raise SystemExit(
        asyncio.run(
            main_async(
                windows=args.windows,
                miner_kind=args.miner,
                as_json=args.json,
                task=args.task,
                source=args.source,
                netuid=args.netuid,
                skip_quality=args.skip_quality,
                cache_index=args.cache_index,
            )
        )
    )
