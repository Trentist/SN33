"""Competitive MinerLib: parallel mine + hybrid tag / skill pipelines with time budgets."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any
from typing import Iterator
from typing import List

import bittensor as bt

from conversationgenome.api.models.raw_metadata import RawMetadata
from conversationgenome.miner.MinerLib import MinerLib
from conversationgenome.task.Task import Task
from custom_miner.parallel_mine import parallel_mine_task
from custom_miner.postprocess import postprocess_result
from custom_miner.prompt_overrides import install_custom_prompts
from custom_miner.prompt_overrides import is_installed
from custom_miner.skill_pipeline import mine_skill_coverage
from custom_miner.tag_pipeline import select_tags_for_task
from custom_miner.timing import TimeBudget

_DETERMINISTIC_COMBINE_TYPES = {
    "conversation_tagging",
    "webpage_metadata_generation",
    "named_entities_extraction",
}


def _deterministic_combine(metadata_tags: list, generateEmbeddings: bool = False) -> RawMetadata | None:
    """Merge tag groups without LLM — enrichment groups first."""
    if not metadata_tags:
        return None

    ordered_groups = list(reversed(metadata_tags))
    merged: List[str] = []
    seen: set[str] = set()
    for group in ordered_groups:
        if not group:
            continue
        tags = group if isinstance(group, list) else list(group)
        for tag in tags:
            if not isinstance(tag, str):
                continue
            norm = tag.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(norm)

    if not merged:
        return None
    return RawMetadata(tags=merged[:24], vectors=None, success=True)


@contextmanager
def _patch_llm_combines() -> Iterator[None]:
    """Patch both metadata and NER combine paths to deterministic merge."""
    import conversationgenome.llm.LlmLib as llmlib_mod
    from conversationgenome.llm.llm_factory import get_llm_backend

    llm = get_llm_backend()
    originals = {
        "combine_metadata_tags": getattr(llm, "combine_metadata_tags", None),
        "combine_named_entities": getattr(llm, "combine_named_entities", None),
    }
    class_originals = {
        "combine_metadata_tags": llmlib_mod.LlmLib.combine_metadata_tags,
        "combine_named_entities": llmlib_mod.LlmLib.combine_named_entities,
    }

    def _bound(metadata_tags, generateEmbeddings=False):
        return _deterministic_combine(metadata_tags, generateEmbeddings=generateEmbeddings)

    llm.combine_metadata_tags = _bound  # type: ignore[method-assign]
    llm.combine_named_entities = _bound  # type: ignore[method-assign]
    llmlib_mod.LlmLib.combine_metadata_tags = (
        lambda self, metadata_tags, generateEmbeddings=False: _deterministic_combine(
            metadata_tags, generateEmbeddings=generateEmbeddings
        )
    )
    llmlib_mod.LlmLib.combine_named_entities = (
        lambda self, named_entities, generateEmbeddings=False: _deterministic_combine(
            named_entities, generateEmbeddings=generateEmbeddings
        )
    )
    try:
        yield
    finally:
        for name, fn in originals.items():
            if fn is not None:
                setattr(llm, name, fn)
        for name, fn in class_originals.items():
            setattr(llmlib_mod.LlmLib, name, fn)


class CompetitiveMinerLib(MinerLib):
    verbose = False

    def __init__(self):
        super().__init__()
        if not is_installed():
            install_custom_prompts()
            bt.logging.info("CompetitiveMinerLib: custom prompt overlay installed")

    async def _mine_tags_raw(self, task: Task) -> dict[str, Any]:
        """Parallel LLM mine when possible; else stock path with deterministic combine."""
        parallel = await parallel_mine_task(task)
        if parallel is not None:
            return parallel

        if task.type in _DETERMINISTIC_COMBINE_TYPES:
            with _patch_llm_combines():
                raw = await task.mine()
        else:
            raw = await task.mine()
        return raw or {"tags": [], "vectors": None}

    async def _mine_named_entities(self, task: Task, budget: TimeBudget) -> dict[str, Any]:
        """Dedicated NER pipeline. The shared tagger pulls this score down."""
        from custom_miner.ner_pipeline import mine_named_entities

        try:
            result = mine_named_entities(task, budget)
        except Exception as exc:
            bt.logging.warning(f"NamedEntityPipeline failed ({exc})")
            result = {"tags": [], "vectors": None}
        tags = list((result or {}).get("tags") or []) if isinstance(result, dict) else []
        if len(tags) >= 3 or budget.remaining() <= 3.0:
            return {"tags": tags, "vectors": None}

        bt.logging.warning("NamedEntityPipeline returned too few tags; falling back to parallel NER extract")
        raw = await parallel_mine_task(task)
        fallback = list((raw or {}).get("tags") or []) if isinstance(raw, dict) else []
        if len(fallback) > len(tags):
            return {"tags": fallback, "vectors": None}
        return {"tags": tags, "vectors": None}

    async def _mine_webpage(self, task: Task, budget: TimeBudget) -> dict[str, Any]:
        """Dedicated webpage pipeline, with the shared tagger as a thin fallback."""
        from conversationgenome.llm.llm_factory import get_llm_backend
        from custom_miner.webpage_pipeline import mine_webpage
        from custom_miner.webpage_pipeline.pipeline import use_webpage_chat_model

        llm = get_llm_backend()
        with use_webpage_chat_model(llm):
            try:
                result = mine_webpage(task, budget)
            except Exception as exc:
                bt.logging.warning(f"WebpagePipeline failed ({exc})")
                result = {"tags": [], "vectors": None}
            tags = list((result or {}).get("tags") or []) if isinstance(result, dict) else []
            if len(tags) >= 3 or budget.remaining() <= 4.0:
                return {"tags": tags, "vectors": None}

            bt.logging.warning("WebpagePipeline returned too few tags; falling back to shared tag pipeline")
            raw = await self._mine_tags_raw(task)
            llm_tags = list((raw or {}).get("tags") or []) if isinstance(raw, dict) else []
            hybrid = select_tags_for_task(task, llm_tags, budget=budget)
            if hybrid:
                return {"tags": hybrid, "vectors": None}
            return postprocess_result(task.type, raw)

    async def do_mining(self, task: Task) -> dict[str, Any]:
        bt.logging.info(f"CompetitiveMiner: mining {task.type}...")
        started = time.perf_counter()
        budget = TimeBudget.for_task(task.type, getattr(task, "timeout", None))

        if task.type == "skill_coverage_evaluation":
            result = mine_skill_coverage(task, budget)
        elif task.type == "skill_generation":
            raw = await task.mine()
            result = postprocess_result(task.type, raw)
        elif task.type == "webpage_metadata_generation":
            result = await self._mine_webpage(task, budget)
        elif task.type == "named_entities_extraction":
            result = await self._mine_named_entities(task, budget)
        else:
            # Overlap LLM mine with local candidate prep kickoff via hybrid after mine
            raw = await self._mine_tags_raw(task)
            llm_tags = list((raw or {}).get("tags") or []) if isinstance(raw, dict) else []
            hybrid = select_tags_for_task(task, llm_tags, budget=budget)
            if hybrid is not None:
                result = {"tags": hybrid, "vectors": None}
                bt.logging.info(
                    f"CompetitiveMiner: hybrid pipeline selected {len(hybrid)} tags "
                    f"(llm_raw={len(llm_tags)} remaining={budget.remaining():.2f}s)"
                )
            else:
                result = postprocess_result(task.type, raw)

        elapsed = time.perf_counter() - started
        tag_count = 0
        n_tests = 0
        if isinstance(result, dict) and task.type == "skill_coverage_evaluation":
            n_tests = sum(len(v) for v in (result.get("section_tests") or {}).values())
            detail = f"sections={len(result.get('section_tests') or {})} tests={n_tests}"
        else:
            tag_count = len(result.get("tags") or []) if isinstance(result, dict) else 0
            detail = f"tags={tag_count}"
        over = " OVER_BUDGET" if elapsed > budget.limit_s else ""
        bt.logging.info(
            f"CompetitiveMiner: finished {task.type} in {elapsed:.2f}s "
            f"(budget={budget.limit_s:.1f}s {detail}){over}"
        )

        try:
            from conversationgenome.extensions.Extensions import Extensions

            Extensions().execute(
                "Metrics",
                "incStat",
                {
                    "metric_name": f"mine_{task.type}",
                    "elapsed": elapsed,
                    "tag_count": tag_count,
                    "test_count": n_tests,
                    "budget_s": budget.limit_s,
                    "over_budget": elapsed > budget.limit_s,
                },
            )
        except Exception:
            pass

        return result
