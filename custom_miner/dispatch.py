"""Route each SN33 task onto the pipeline built for that task.

Pipelines are synchronous. ``CompetitiveMinerLib`` runs ``mine_task`` on a
worker thread so two or three axon requests can proceed together.
"""

from __future__ import annotations

import asyncio
from typing import Any

import bittensor as bt

from custom_miner.ner_pipeline import mine_named_entities
from custom_miner.parallel_mine import parallel_mine_task
from custom_miner.postprocess import postprocess_result
from custom_miner.skill_pipeline import mine_skill_coverage
from custom_miner.tag_pipeline import select_tags_for_task
from custom_miner.timing import TimeBudget
from custom_miner.webpage_pipeline import mine_webpage

# task.type -> pipeline name logged with each result
PIPELINES: dict[str, str] = {
    "conversation_tagging": "tag",
    "survey_tagging": "tag",
    "webpage_metadata_generation": "webpage",
    "named_entities_extraction": "ner",
    "skill_coverage_evaluation": "skill_coverage",
    "skill_generation": "skill_generation",
}


def pipeline_for(task_type: str) -> str:
    return PIPELINES.get(task_type or "", "stock")


def empty_result(task_type: str) -> dict[str, Any]:
    if task_type == "skill_coverage_evaluation":
        return {"skill": "", "tdd_plan": "", "section_tests": {}}
    return {"tags": [], "vectors": None}


def _run_async(coro):
    """Run a stock async ``task.mine()`` from a worker thread."""
    return asyncio.run(coro)


def _tags_of(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    tags = raw.get("tags") or []
    return [t for t in tags if isinstance(t, str)]


def _mine_tags(task, budget: TimeBudget) -> dict[str, Any]:
    raw = parallel_mine_task(task)
    if raw is None:
        raw = _run_async(task.mine())
    llm_tags = _tags_of(raw)
    selected = select_tags_for_task(task, llm_tags, budget=budget)
    if selected is not None:
        bt.logging.info(
            f"TagPipeline: selected {len(selected)} tags "
            f"(llm_raw={len(llm_tags)} remaining={budget.remaining():.2f}s)"
        )
        return {"tags": selected, "vectors": None}
    return postprocess_result(task.type, raw if isinstance(raw, dict) else None)


def _mine_webpage(task, budget: TimeBudget) -> dict[str, Any]:
    try:
        result = mine_webpage(task, budget)
    except Exception as exc:
        bt.logging.warning(f"WebpagePipeline failed ({exc})")
        result = {"tags": [], "vectors": None}
    tags = _tags_of(result)
    if len(tags) >= 3 or budget.remaining() <= 4.0:
        return {"tags": tags, "vectors": None}

    bt.logging.warning("WebpagePipeline returned too few tags; falling back to tag pipeline")
    raw = parallel_mine_task(task) or {"tags": [], "vectors": None}
    selected = select_tags_for_task(task, _tags_of(raw), budget=budget)
    if selected:
        return {"tags": selected, "vectors": None}
    return postprocess_result(task.type, raw)


def _mine_ner(task, budget: TimeBudget) -> dict[str, Any]:
    try:
        result = mine_named_entities(task, budget)
    except Exception as exc:
        bt.logging.warning(f"NamedEntityPipeline failed ({exc})")
        result = {"tags": [], "vectors": None}
    tags = _tags_of(result)
    if len(tags) >= 3 or budget.remaining() <= 3.0:
        return {"tags": tags, "vectors": None}

    bt.logging.warning("NamedEntityPipeline returned too few tags; falling back to parallel NER extract")
    raw = parallel_mine_task(task) or {"tags": [], "vectors": None}
    fallback = _tags_of(raw)
    if len(fallback) > len(tags):
        return {"tags": fallback, "vectors": None}
    return {"tags": tags, "vectors": None}


def _mine_skill_generation(task, budget: TimeBudget) -> dict[str, Any]:
    if budget.expired():
        return empty_result(task.type)
    raw = _run_async(task.mine())
    return postprocess_result(task.type, raw if isinstance(raw, dict) else None)


def mine_task(task, budget: TimeBudget) -> dict[str, Any]:
    """Run the pipeline registered for ``task.type``. Safe to call off the event loop."""
    task_type = getattr(task, "type", "") or ""
    name = pipeline_for(task_type)
    if budget.expired():
        bt.logging.warning(f"Dispatch: {task_type} skipped, budget already spent")
        return empty_result(task_type)

    if name == "skill_coverage":
        result = mine_skill_coverage(task, budget)
    elif name == "webpage":
        result = _mine_webpage(task, budget)
    elif name == "ner":
        result = _mine_ner(task, budget)
    elif name == "skill_generation":
        result = _mine_skill_generation(task, budget)
    elif name == "tag":
        result = _mine_tags(task, budget)
    else:
        bt.logging.warning(f"Dispatch: no pipeline for {task_type!r}; using stock mine")
        raw = _run_async(task.mine())
        result = postprocess_result(task_type, raw if isinstance(raw, dict) else None)

    if not isinstance(result, dict):
        return empty_result(task_type)
    return result
