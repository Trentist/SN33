"""Webpage metadata miner.

Mirrors validator ground truth (stock page, snippet, and combine prompts),
asks for synonyms of those combine tags, then keeps a score-shaped portfolio:

  up to 5 combine-order overlaps  +  up to 5 paraphrases near that centroid
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
from custom_miner.timing import TimeBudget
from custom_miner.webpage_pipeline.anchors import phrases_in_text
from custom_miner.webpage_pipeline.anchors import variant_tags
from custom_miner.webpage_pipeline.gate import clean_tags
from custom_miner.webpage_pipeline.parse import is_stub_page
from custom_miner.webpage_pipeline.parse import parse_webpage_task
from custom_miner.webpage_pipeline.portfolio import _PERIPHERAL
from custom_miner.webpage_pipeline.portfolio import embed_tags
from custom_miner.webpage_pipeline.portfolio import lexical_mix
from custom_miner.webpage_pipeline.portfolio import select_portfolio
from custom_miner.webpage_pipeline.stock import combine_prompt
from custom_miner.webpage_pipeline.stock import enrichment_prompt
from custom_miner.webpage_pipeline.stock import neighborhood_prompt
from custom_miner.webpage_pipeline.stock import paraphrase_prompt
from custom_miner.webpage_pipeline.stock import parse_tag_response
from custom_miner.webpage_pipeline.stock import website_prompt

_COMBINE_MIN_REMAINING = 3.5
_NEIGHBORHOOD_MIN_REMAINING = 1.8
_PARAPHRASE_MIN_REMAINING = 3.5
_DEFAULT_WEBPAGE_MODEL = "google/gemini-2.5-flash"


def webpage_chat_model() -> str:
    """Chat model for webpage tag generation. Embeddings stay on the validator model."""
    return (
        os.environ.get("WEBPAGE_OPENROUTER_MODEL", _DEFAULT_WEBPAGE_MODEL).strip()
        or _DEFAULT_WEBPAGE_MODEL
    )


@contextmanager
def use_webpage_chat_model(llm: Any) -> Iterator[Any]:
    """Pin chat completions to Gemini Flash for this webpage mine only."""
    previous = getattr(llm, "model", None)
    if hasattr(llm, "model"):
        llm.model = webpage_chat_model()
    try:
        yield llm
    finally:
        if previous is not None and hasattr(llm, "model"):
            llm.model = previous


def _run_jobs(jobs: list[Callable[[], Any]], max_workers: int = 6) -> list[Any]:
    if not jobs:
        return []
    if len(jobs) == 1:
        try:
            return [jobs[0]()]
        except Exception as exc:
            bt.logging.warning(f"WebpagePipeline job failed: {exc}")
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
                bt.logging.warning(f"WebpagePipeline job {idx} failed: {exc}")
                out[idx] = None
    return out


def _complete(llm: Any, prompt: str, budget: TimeBudget | None = None) -> list[str]:
    for attempt in range(3):
        try:
            raw = llm.basic_prompt(prompt)
        except Exception as exc:
            bt.logging.warning(f"WebpagePipeline prompt failed: {exc}")
            raw = None
        tags = parse_tag_response(raw if isinstance(raw, str) else None)
        wait = 2.0 * (attempt + 1)
        if tags or attempt == 2 or budget is None or budget.remaining() < wait + 1.5:
            return tags
        time.sleep(wait)
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


def _extract(
    llm: Any, page: str, snippets: list[str], *, coding: bool, budget: TimeBudget | None = None
) -> tuple[list[str], list[list[str]]]:
    jobs: list[Callable[[], list[str]]] = []
    page_offset = 0
    if page.strip() and not is_stub_page(page):
        jobs.append(lambda: _complete(llm, website_prompt(page, coding=coding), budget))
        page_offset = 1
    for snippet in snippets:
        if snippet.strip():
            jobs.append(lambda text=snippet: _complete(llm, enrichment_prompt(text, coding=coding), budget))
    results = _run_jobs(jobs)
    lists = [tags if isinstance(tags, list) else [] for tags in results]
    page_tags = lists[0] if page_offset else []
    snippet_groups = lists[page_offset:]
    return page_tags, snippet_groups


def _combine_and_neighbors(
    llm: Any,
    page: str,
    snippets: list[str],
    page_tags: list[str],
    snippet_groups: list[list[str]],
    budget: TimeBudget,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(combine_tags, leftover_tags, neighborhood)``."""
    snippet_tags = _merge_groups(snippet_groups, limit=20)
    base = _merge_groups([page_tags, snippet_tags], limit=15)
    if not base and not page.strip():
        return [], [], []

    remaining = budget.remaining()
    jobs: list[Callable[[], tuple[str, list[str]]]] = []
    nonempty = [group for group in [page_tags, *snippet_groups] if group]
    if nonempty and remaining >= _COMBINE_MIN_REMAINING:
        prompt = combine_prompt(nonempty)

        def _combine(prompt=prompt) -> tuple[str, list[str]]:
            return ("combine", _complete(llm, prompt, budget))

        jobs.append(_combine)
    if remaining >= _NEIGHBORHOOD_MIN_REMAINING and (page.strip() or base):
        prompt = neighborhood_prompt(page, snippets, base)

        def _neighborhood(prompt=prompt) -> tuple[str, list[str]]:
            return ("neighborhood", _complete(llm, prompt, budget))

        jobs.append(_neighborhood)

    combined: list[str] = []
    neighbors: list[str] = []
    for kind, tags in [item for item in _run_jobs(jobs) if isinstance(item, tuple)]:
        if kind == "combine":
            combined = tags
        elif kind == "neighborhood":
            neighbors = tags

    if combined:
        extra = [tag for tag in base if tag not in set(combined)]
        return combined[:15], extra, neighbors
    return base, [], neighbors


def mine_webpage(task, budget: TimeBudget | None = None) -> dict[str, Any]:
    """Run the webpage metadata pipeline. Returns ``{tags, vectors}``."""
    if budget is None:
        budget = TimeBudget.for_task("webpage_metadata_generation")

    page, snippets, coding = parse_webpage_task(task)
    if not page and not snippets:
        return {"tags": [], "vectors": None}
    page_for_model = "" if is_stub_page(page) else page
    evidenced = phrases_in_text("\n".join([page_for_model, *snippets]))

    llm = get_llm_backend()
    with use_webpage_chat_model(llm):
        page_tags, snippet_groups = _extract(
            llm, page_for_model, snippets, coding=coding, budget=budget
        )
        combined, extras, neighbors = _combine_and_neighbors(
            llm, page_for_model, snippets, page_tags, snippet_groups, budget
        )
        if combined or extras:
            core_hits = [tag for tag in evidenced if tag not in _PERIPHERAL]
            combined_core = [tag for tag in combined if tag not in _PERIPHERAL]
            extra_core = [tag for tag in extras if tag not in _PERIPHERAL]
            tail = [tag for tag in combined + extras if tag in _PERIPHERAL]
            centroid_tags = clean_tags(combined_core + core_hits + extra_core)
            paraphrases: list[str] = []
            if centroid_tags and budget.remaining() >= _PARAPHRASE_MIN_REMAINING:
                paraphrases = _complete(llm, paraphrase_prompt(centroid_tags[:8]), budget)
            gt = clean_tags(centroid_tags + tail)
            neighborhood = clean_tags(variant_tags(centroid_tags) + paraphrases + neighbors)
        else:
            centroid_tags = clean_tags([tag for tag in evidenced if tag not in _PERIPHERAL])
            gt = clean_tags(centroid_tags + [tag for tag in evidenced if tag in _PERIPHERAL])
            neighborhood = clean_tags(variant_tags(centroid_tags or evidenced))

        used_embed = False
        if budget.allow_embed() and (gt or neighborhood):
            vectors = embed_tags(llm, gt + neighborhood)
            if vectors:
                selected = select_portfolio(
                    gt, neighborhood, vectors, centroid_from=centroid_tags
                )
                used_embed = True
            else:
                selected = lexical_mix(gt, neighborhood)
        else:
            selected = lexical_mix(gt, neighborhood)

    selected = clean_tags(selected)[:10]
    bt.logging.info(
        f"WebpagePipeline: model={webpage_chat_model()} coding={coding} snippets={len(snippets)} "
        f"gt={len(gt)} neighborhood={len(neighborhood)} selected={len(selected)} "
        f"embed={used_embed} tags={selected}"
    )
    return {"tags": selected, "vectors": None}
