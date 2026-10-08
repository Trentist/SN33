"""Named-entity miner.

Mirrors validator ground truth (stock transcript, enrichment, and combine
prompts), restores the surface forms that combine collapses, then keeps a
centroid portfolio:

  likely exact entity strings  +  high-cosine aliases and sibling entities
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from contextlib import contextmanager
from typing import Any
from typing import Callable
from typing import Iterator

import bittensor as bt

from conversationgenome.llm.llm_factory import get_llm_backend
from custom_miner.ner_pipeline.aliases import anchor_paraphrases
from custom_miner.ner_pipeline.aliases import evidenced
from custom_miner.ner_pipeline.aliases import grounded_aliases
from custom_miner.ner_pipeline.aliases import recovered_entities
from custom_miner.ner_pipeline.gate import clean_tags
from custom_miner.ner_pipeline.parse import parse_ner_task
from custom_miner.ner_pipeline.portfolio import embed_tags
from custom_miner.ner_pipeline.portfolio import lexical_mix
from custom_miner.ner_pipeline.portfolio import select_portfolio
from custom_miner.ner_pipeline.stock import combine_prompt
from custom_miner.ner_pipeline.stock import enrichment_prompt
from custom_miner.ner_pipeline.stock import neighborhood_prompt
from custom_miner.ner_pipeline.stock import parse_tag_response
from custom_miner.ner_pipeline.stock import transcript_prompt
from custom_miner.timing import TimeBudget

_COMBINE_MIN_REMAINING = 3.5
_NEIGHBORHOOD_MIN_REMAINING = 1.8
_DEFAULT_NER_MODEL = "anthropic/claude-haiku-5.5"


def ner_chat_model() -> str:
    """Chat model for entity extraction. Embeddings stay on text-embedding-3-small.

    ``anthropic/claude-haiku-5.5`` is the default. ``NER_OPENROUTER_MODEL`` overrides it.
    The miner-wide Mercury model is left unchanged for other task types.
    """
    return os.environ.get("NER_OPENROUTER_MODEL", "").strip() or _DEFAULT_NER_MODEL


@contextmanager
def use_ner_chat_model(llm: Any) -> Iterator[Any]:
    """Pin this mine to the NER chat model and skip reasoning."""
    previous = getattr(llm, "model", None)
    previous_effort = getattr(llm, "reasoning_effort", None)
    if hasattr(llm, "model"):
        llm.model = ner_chat_model()
    if hasattr(llm, "reasoning_effort"):
        llm.reasoning_effort = None
    try:
        yield llm
    finally:
        if previous is not None and hasattr(llm, "model"):
            llm.model = previous
        if hasattr(llm, "reasoning_effort"):
            llm.reasoning_effort = previous_effort


def _run_jobs(jobs: list[Callable[[], Any]], max_workers: int = 6) -> list[Any]:
    if not jobs:
        return []
    if len(jobs) == 1:
        try:
            return [jobs[0]()]
        except Exception as exc:
            bt.logging.warning(f"NamedEntityPipeline job failed: {exc}")
            return [None]
    workers = min(max_workers, len(jobs))
    out: list[Any] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {pool.submit(fn): i for i, fn in enumerate(jobs)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                out[idx] = fut.result()
            except Exception as exc:
                bt.logging.warning(f"NamedEntityPipeline job {idx} failed: {exc}")
                out[idx] = None
    return out


def _complete(llm: Any, prompt: str) -> list[str]:
    for attempt in range(2):
        try:
            raw = llm.basic_prompt(prompt)
        except Exception as exc:
            bt.logging.warning(f"NamedEntityPipeline prompt failed: {exc}")
            raw = None
        tags = parse_tag_response(raw if isinstance(raw, str) else None)
        if tags or attempt == 1:
            return tags
        time.sleep(0.4)
    return []


def _merge_groups(groups: list[list[str]], limit: int = 15) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for tag in group:
            if tag in seen:
                continue
            seen.add(tag)
            merged.append(tag)
            if len(merged) >= limit:
                return merged
    return merged


def _extract(llm: Any, transcript: str, snippets: list[str]) -> tuple[list[str], list[list[str]]]:
    jobs: list[Callable[[], list[str]]] = []
    transcript_offset = 0
    if transcript.strip():
        jobs.append(lambda: _complete(llm, transcript_prompt(transcript)))
        transcript_offset = 1
    for snippet in snippets:
        if snippet.strip():
            jobs.append(lambda text=snippet: _complete(llm, enrichment_prompt(text)))
    results = _run_jobs(jobs)
    lists = [tags if isinstance(tags, list) else [] for tags in results]
    transcript_tags = lists[0] if transcript_offset else []
    snippet_groups = lists[transcript_offset:]
    return transcript_tags, snippet_groups


def _expand(
    llm: Any,
    transcript: str,
    snippets: list[str],
    transcript_tags: list[str],
    snippet_groups: list[list[str]],
    budget: TimeBudget,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return ``(entity pool, close aliases, bridges, centroid core)``."""
    snippet_tags = _merge_groups(snippet_groups, limit=20)
    base = _merge_groups([transcript_tags, snippet_tags], limit=15)
    source = "\n".join([transcript, *snippets])
    if not base and not transcript.strip():
        return [], [], [], []

    remaining = budget.remaining()
    jobs: list[Callable[[], tuple[str, list[str]]]] = []
    nonempty = [group for group in [transcript_tags, *snippet_groups] if group]
    if nonempty and remaining >= _COMBINE_MIN_REMAINING:
        prompt = combine_prompt(nonempty)

        def _combine(prompt=prompt) -> tuple[str, list[str]]:
            return ("combine", _complete(llm, prompt))

        jobs.append(_combine)
    if remaining >= _NEIGHBORHOOD_MIN_REMAINING and (transcript.strip() or base):
        prompt = neighborhood_prompt(transcript, snippets, base)

        def _neighborhood(prompt=prompt) -> tuple[str, list[str]]:
            return ("neighborhood", _complete(llm, prompt))

        jobs.append(_neighborhood)

    combined: list[str] = []
    neighbors: list[str] = []
    for kind, tags in [item for item in _run_jobs(jobs) if isinstance(item, tuple)]:
        if kind == "combine":
            combined = tags
        elif kind == "neighborhood":
            neighbors = tags

    if combined:
        core = list(combined[:15])
        extra = [tag for tag in transcript_tags + snippet_tags if tag not in set(core)]
    else:
        core = list(base)
        extra = []
    core = [tag for tag in clean_tags(core) if evidenced(tag, source)]
    extra = [tag for tag in clean_tags(extra) if evidenced(tag, source) and tag not in set(core)]
    recovered = [
        tag
        for tag in recovered_entities(source, set(core) | set(extra))
        if evidenced(tag, source)
    ]
    # Centroid is the model list. Recovered names are candidates, not the center.
    pool = clean_tags(core + extra + recovered)
    if not pool:
        pool = [tag for tag in clean_tags(grounded_aliases(source, base)) if evidenced(tag, source)][:8]
        core = pool
    aliases = [
        tag
        for tag in grounded_aliases(source, core or pool) + anchor_paraphrases(core or pool, source)
        if tag not in set(pool) and evidenced(tag, source)
    ]
    bridges = [
        tag
        for tag in clean_tags(neighbors)
        if tag not in set(pool) and tag not in set(aliases) and evidenced(tag, source)
    ][:3]
    return pool, clean_tags(aliases), bridges, core


def mine_named_entities(task, budget: TimeBudget | None = None) -> dict[str, Any]:
    """Run the named-entity pipeline. Returns ``{tags, vectors}``."""
    if budget is None:
        budget = TimeBudget.for_task("named_entities_extraction")

    transcript, snippets = parse_ner_task(task)
    if not transcript and not snippets:
        return {"tags": [], "vectors": None}

    llm = get_llm_backend()
    with use_ner_chat_model(llm):
        transcript_tags, snippet_groups = _extract(llm, transcript, snippets)
        gt, aliases, bridges, core = _expand(
            llm, transcript, snippets, transcript_tags, snippet_groups, budget
        )

    used_embed = False
    # One embedding batch is what keeps the set on the centroid. Skipping it
    # falls back to a person list and the 55% term collapses.
    if budget.remaining() > 0.3 and (gt or aliases or bridges):
        embed_list = []
        embed_seen: set[str] = set()
        for tag in list(core) + list(gt) + list(aliases) + list(bridges):
            if tag and tag not in embed_seen:
                embed_seen.add(tag)
                embed_list.append(tag)
            if len(embed_list) >= 24:
                break
        vectors = embed_tags(llm, embed_list)
        if vectors:
            selected = select_portfolio(gt, aliases, vectors, bridges=bridges, core=core)
            used_embed = True
        else:
            selected = lexical_mix(core or gt, aliases + bridges)
    else:
        selected = lexical_mix(core or gt, aliases + bridges)

    selected = clean_tags(selected)[:10]
    bt.logging.info(
        f"NamedEntityPipeline: model={ner_chat_model()} snippets={len(snippets)} "
        f"pool={len(gt)} aliases={len(aliases)} bridges={len(bridges)} "
        f"selected={len(selected)} embed={used_embed} tags={selected}"
    )
    return {"tags": selected, "vectors": None}
