"""Named-entity miner.

Returns three tags. Anchors are read from the transcript and snippets. The
selector scores each triple against simulated ground truths with the validator
formula, including exact-match exclusion and zero-padding. Chat extraction
does not define the centroid.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any
from typing import Iterator

import bittensor as bt

from conversationgenome.llm.llm_factory import get_llm_backend
from custom_miner.ner_pipeline.gate import clean_tags
from custom_miner.ner_pipeline.parse import parse_ner_task
from custom_miner.ner_pipeline.portfolio import embed_tags
from custom_miner.ner_pipeline.triple import candidate_pool
from custom_miner.ner_pipeline.triple import document_seed
from custom_miner.ner_pipeline.triple import prepare_document
from custom_miner.ner_pipeline.triple import select_triple
from custom_miner.ner_pipeline.triple import simulate_ground_truths
from custom_miner.timing import TimeBudget

_DEFAULT_NER_MODEL = "anthropic/claude-haiku-5.5"
# One embedding request. A second batch, or a 2s/4s/6s retry, is what crosses 12s.
_EMBED_CAP = 32
_EMBED_TIMEOUT_S = 3.5


def ner_chat_model() -> str:
    """Optional NER chat model. Embeddings stay on text-embedding-3-small.

    ``NER_OPENROUTER_MODEL`` overrides the default. The miner-wide model is
    left unchanged for other task types. The three-tag selector does not call it.
    """
    return os.environ.get("NER_OPENROUTER_MODEL", "").strip() or _DEFAULT_NER_MODEL


@contextmanager
def use_ner_chat_model(llm: Any) -> Iterator[Any]:
    """Pin a backend to the NER chat model and skip reasoning."""
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


def mine_named_entities(task, budget: TimeBudget | None = None) -> dict[str, Any]:
    """Return the triple with the best exact-aware simulated score."""
    started = time.perf_counter()
    if budget is None:
        budget = TimeBudget.for_task("named_entities_extraction")

    transcript, snippets = parse_ner_task(task)
    if not transcript and not snippets:
        return {"tags": [], "vectors": None}

    transcript_anchors, groups, bridges = prepare_document(transcript, snippets)
    anchors: list[str] = []
    seen: set[str] = set()
    for tag in transcript_anchors + [tag for group in groups for tag in group]:
        if tag and tag not in seen:
            seen.add(tag)
            anchors.append(tag)
    pool = candidate_pool(anchors, bridges)
    if len(pool) < 3:
        return {"tags": clean_tags(anchors + bridges)[:3], "vectors": None}

    embed_list: list[str] = []
    embed_seen: set[str] = set()
    for tag in list(pool) + list(transcript_anchors) + anchors:
        if tag and tag not in embed_seen:
            embed_seen.add(tag)
            embed_list.append(tag)
        if len(embed_list) >= _EMBED_CAP:
            break

    llm = get_llm_backend()
    vectors = {}
    if budget.remaining() > 0.3:
        vectors = embed_tags(
            llm,
            embed_list,
            timeout=min(_EMBED_TIMEOUT_S, max(0.8, budget.remaining() - 0.4)),
            attempts=2 if budget.remaining() > 5.0 else 1,
            retry_sleep=0.15,
        )
    choice = None
    if vectors:
        simulations = simulate_ground_truths(
            transcript_anchors,
            groups,
            seed=document_seed(transcript, snippets),
        )
        choice = select_triple(pool, vectors, simulations)
    selected = list(choice.tags) if choice is not None else clean_tags(pool)[:3]
    bt.logging.info(
        f"NamedEntityPipeline: anchors={len(anchors)} bridges={len(bridges)} "
        f"embedded={len(embed_list)} selected={selected} "
        f"objective={None if choice is None else round(choice.objective, 3)} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    return {"tags": selected, "vectors": None}
