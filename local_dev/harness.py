"""Offline validator↔miner simulation harness (no chain required)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace
from typing import Any
from typing import Literal

from conversationgenome.miner.MinerLib import MinerLib
from conversationgenome.task.Task import Task
from conversationgenome.task_bundle.TaskBundle import TaskBundle
from local_dev.fixtures import attach_pseudo_skill_scoring
from local_dev.fixtures import attach_pseudo_vectors
from local_dev.fixtures import ready_conversation_bundle
from local_dev.fixtures import ready_skill_coverage_bundle
from local_dev.mock_llm import mock_mine

MinerKind = Literal["baseline", "custom"]
Mode = Literal["mock", "live"]
TaskSim = Literal["conversation_tagging", "skill_coverage_evaluation"]


@dataclass
class SimResult:
    miner: MinerKind
    mode: Mode
    task_type: str
    tags: list[str]
    adjusted_score: float
    final_miner_score: float
    elapsed_s: float
    section_coverage: float | None = None
    skill_coverage: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_miner_lib(kind: MinerKind) -> MinerLib:
    if kind == "custom":
        from custom_miner.miner_lib import CompetitiveMinerLib

        return CompetitiveMinerLib()
    return MinerLib()


async def _mine_one(task: Task, kind: MinerKind, mode: Mode) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if mode == "mock":
        raw = await mock_mine(task)
        if kind == "custom" and task.type != "skill_coverage_evaluation":
            from custom_miner.postprocess import postprocess_result

            raw = postprocess_result(task.type, raw)
        elif kind == "custom" and task.type == "skill_coverage_evaluation":
            from custom_miner.skill_pipeline import normalize_skill_bundle

            section_map = getattr(getattr(task.input, "data", None), "section_map", None)
            raw = normalize_skill_bundle(raw, section_map)
    else:
        ml = _get_miner_lib(kind)
        raw = await ml.do_mining(task)
    return raw, time.perf_counter() - started


async def run_conversation_sim(
    *,
    miner: MinerKind = "custom",
    mode: Mode = "mock",
    window_limit: int = 2,
) -> list[SimResult]:
    """
    Reserve a local conversation bundle, mine each window, score vs ground truth.

    mock mode: keyword miner + pseudo-embeddings (no API key)
    live mode: real LLM via MinerLib/CompetitiveMinerLib (needs OPENAI_API_KEY)
               still uses pseudo-embeddings unless USE_REAL_EMBEDDINGS=1
    """
    bundle: TaskBundle = ready_conversation_bundle()
    tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=window_limit)
    results: list[SimResult] = []

    use_real_embeddings = os.environ.get("USE_REAL_EMBEDDINGS", "0").strip() in ("1", "true", "True")

    for task in tasks:
        raw, elapsed = await _mine_one(task, miner, mode)

        if use_real_embeddings and mode == "live":
            formatted = await bundle.format_results(raw)
        else:
            formatted = attach_pseudo_vectors(raw)

        # Evaluate expects objects with .cgp_output / .axon (not plain dicts)
        response = SimpleNamespace(
            cgp_output=[formatted],
            axon=SimpleNamespace(uuid="local", hotkey=miner),
        )
        final_scores, _rank = await bundle.evaluate(miner_responses=[response])
        entry = (final_scores or [{}])[0]

        results.append(
            SimResult(
                miner=miner,
                mode=mode,
                task_type=task.type,
                tags=list(formatted.get("tags") or []),
                adjusted_score=float(entry.get("adjustedScore") or 0.0),
                final_miner_score=float(entry.get("final_miner_score") or 0.0),
                elapsed_s=elapsed,
            )
        )

    return results


async def run_skill_coverage_sim(
    *,
    miner: MinerKind = "custom",
    mode: Mode = "mock",
    window_limit: int = 1,
) -> list[SimResult]:
    """
    Offline / live skill_coverage_evaluation cycle.

    mock: deterministic skill bundle + pseudo judge/embeddings (no API key)
    live: CompetitiveMinerLib/MinerLib skill pipeline; USE_REAL_EMBEDDINGS=1
          runs full format_results (judge + embeddings)
    """
    bundle: TaskBundle = ready_skill_coverage_bundle()
    tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=window_limit)
    results: list[SimResult] = []
    use_real = os.environ.get("USE_REAL_EMBEDDINGS", "0").strip() in ("1", "true", "True")
    section_vectors = getattr(getattr(bundle.input, "metadata", None), "section_vectors", None) or {}

    for task in tasks:
        raw, elapsed = await _mine_one(task, miner, mode)

        if use_real and mode == "live":
            formatted = await bundle.format_results(dict(raw))
        else:
            formatted = attach_pseudo_skill_scoring(dict(raw), section_vectors)

        response = SimpleNamespace(
            cgp_output=[formatted],
            axon=SimpleNamespace(uuid="local", hotkey=miner),
        )
        final_scores, _rank = await bundle.evaluate(miner_responses=[response])
        entry = (final_scores or [{}])[0]
        section_keys = list((formatted.get("section_tests") or {}).keys())

        results.append(
            SimResult(
                miner=miner,
                mode=mode,
                task_type=task.type,
                tags=section_keys,
                adjusted_score=float(entry.get("adjustedScore") or 0.0),
                final_miner_score=float(entry.get("final_miner_score") or 0.0),
                elapsed_s=elapsed,
                section_coverage=entry.get("section_coverage"),
                skill_coverage=entry.get("skill_coverage"),
                extra={
                    "n_tests": sum(len(v) for v in (formatted.get("section_tests") or {}).values()),
                    "skill_chars": len(formatted.get("skill") or ""),
                },
            )
        )

    return results


async def compare_miners(mode: Mode = "mock") -> dict[str, list[SimResult]]:
    baseline = await run_conversation_sim(miner="baseline", mode=mode)
    custom = await run_conversation_sim(miner="custom", mode=mode)
    return {"baseline": baseline, "custom": custom}


async def compare_skill_miners(mode: Mode = "mock") -> dict[str, list[SimResult]]:
    baseline = await run_skill_coverage_sim(miner="baseline", mode=mode)
    custom = await run_skill_coverage_sim(miner="custom", mode=mode)
    return {"baseline": baseline, "custom": custom}


def print_results(results: list[SimResult]) -> None:
    for r in results:
        extra = ""
        if r.section_coverage is not None or r.skill_coverage is not None:
            extra = (
                f" sec={r.section_coverage if r.section_coverage is not None else '-'} "
                f"skill={r.skill_coverage if r.skill_coverage is not None else '-'}"
            )
        print(
            f"[{r.miner}/{r.mode}] score={r.final_miner_score:.4f} "
            f"adj={r.adjusted_score:.4f} out={len(r.tags)} time={r.elapsed_s:.2f}s{extra}"
        )
        print(f"  keys: {', '.join(str(t) for t in r.tags[:15])}")


def main_compare_sync(mode: Mode = "mock") -> None:
    data = asyncio.run(compare_miners(mode=mode))
    print("=== baseline ===")
    print_results(data["baseline"])
    print("=== custom ===")
    print_results(data["custom"])
    print("=== json ===")
    print(json.dumps({k: [x.to_dict() for x in v] for k, v in data.items()}, indent=2))
