"""Parallel stock-equivalent mining for multi-LLM-call task types.

Runs independent LLM tag extractions concurrently via a thread pool (OpenAI /
OpenRouter clients are sync), then merges with deterministic combine.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Callable

import bittensor as bt


def _tags_of(result) -> list[str]:
    if result is None:
        return []
    tags = getattr(result, "tags", None)
    if tags:
        return list(tags)
    if isinstance(result, dict):
        return list(result.get("tags") or [])
    return []


def _run_jobs(jobs: list[Callable[[], Any]], max_workers: int = 6) -> list[Any]:
    if not jobs:
        return []
    if len(jobs) == 1:
        return [jobs[0]()]
    workers = min(max_workers, len(jobs))
    out: list[Any] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {pool.submit(fn): i for i, fn in enumerate(jobs)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                bt.logging.warning(f"parallel mine job {i} failed: {e}")
                out[i] = None
    return out


def _merge_tag_groups(groups: list[list[str]]) -> dict[str, Any]:
    """Deterministic merge: later groups (enrichment) first, then primary."""
    non_empty = [g for g in groups if g]
    if not non_empty:
        return {"tags": [], "vectors": None}
    ordered = list(reversed(non_empty))
    merged: list[str] = []
    seen: set[str] = set()
    for g in ordered:
        for t in g:
            if not isinstance(t, str):
                continue
            n = t.strip().lower()
            if not n or n in seen:
                continue
            seen.add(n)
            merged.append(n)
    return {"tags": merged[:24], "vectors": None}


def parallel_mine_conversation(task) -> dict[str, Any]:
    from conversationgenome.api.models.conversation import Conversation
    from conversationgenome.llm.llm_factory import get_llm_backend

    llml = get_llm_backend()
    data = task.input.data
    categories = getattr(task.input, "input_categories", None)
    prompt = None
    try:
        prompt = task.prompt_chain[0].prompt_template
    except Exception:
        prompt = None

    conversation = Conversation(
        guid=task.input.guid,
        lines=data.window,
        miner_task_prompt=prompt,
        input_categories=categories,
    )
    enrichment_lines = list(getattr(data, "enrichment_lines", None) or [])

    jobs: list[Callable[[], Any]] = [
        lambda: llml.conversation_to_metadata(conversation=conversation, generateEmbeddings=False)
    ]
    for _, content in enrichment_lines:
        jobs.append(
            lambda c=content: llml.enrichment_to_metadata(
                c, generateEmbeddings=False, input_categories=categories
            )
        )

    results = _run_jobs(jobs)
    groups = [_tags_of(r) for r in results]
    return _merge_tag_groups(groups)


def parallel_mine_ner(task) -> dict[str, Any]:
    from conversationgenome.llm.llm_factory import get_llm_backend

    llml = get_llm_backend()
    window = list(getattr(task.input.data, "window", None) or [])
    if not window:
        return {"tags": [], "vectors": None}

    jobs: list[Callable[[], Any]] = []
    for idx, (_line_idx, content) in enumerate(window):
        if idx == 0:
            jobs.append(lambda c=content: llml.raw_transcript_to_named_entities(c, generateEmbeddings=False))
        else:
            jobs.append(lambda c=content: llml.enrichment_to_NER(c, generateEmbeddings=False))

    results = _run_jobs(jobs)
    groups = [_tags_of(r) for r in results]
    return _merge_tag_groups(groups)


def parallel_mine_webpage(task) -> dict[str, Any]:
    from conversationgenome.llm.llm_factory import get_llm_backend

    llml = get_llm_backend()
    categories = getattr(task.input, "input_categories", None)
    window = list(getattr(task.input.data, "window", None) or [])
    if not window:
        return {"tags": [], "vectors": None}

    jobs: list[Callable[[], Any]] = []
    for idx, (_line_idx, content) in enumerate(window):
        if idx == 0:
            jobs.append(
                lambda c=content: llml.website_to_metadata(
                    c, generateEmbeddings=False, input_categories=categories
                )
            )
        else:
            jobs.append(
                lambda c=content: llml.enrichment_to_metadata(
                    c, generateEmbeddings=False, input_categories=categories
                )
            )

    results = _run_jobs(jobs)
    groups = [_tags_of(r) for r in results]
    return _merge_tag_groups(groups)


async def parallel_mine_task(task) -> dict[str, Any] | None:
    """Return mined dict for supported types, or None to fall back to task.mine()."""
    t = getattr(task, "type", "")
    if t == "conversation_tagging":
        return parallel_mine_conversation(task)
    if t == "named_entities_extraction":
        return parallel_mine_ner(task)
    if t == "webpage_metadata_generation":
        return parallel_mine_webpage(task)
    return None
