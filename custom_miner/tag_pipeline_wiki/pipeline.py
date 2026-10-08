"""WikiGround end-to-end selector.

conversation → segment → mentions → Wikidata link → expand → gate → select
"""

from __future__ import annotations

import os

import bittensor as bt

from custom_miner.policies import get_tag_policy
from custom_miner.tag_pipeline.text_extract import extract_task_texts
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import normalize_tag
from custom_miner.tag_pipeline_wiki.expand import expand_entities
from custom_miner.tag_pipeline_wiki.gate import gate_candidates
from custom_miner.tag_pipeline_wiki.link import link_mentions
from custom_miner.tag_pipeline_wiki.mentions import extract_mentions
from custom_miner.tag_pipeline_wiki.segment import segment_dialogue
from custom_miner.tag_pipeline_wiki.select import select_tags
from custom_miner.timing import TimeBudget

_FINAL_JUNK = {
    "owners", "converting", "capital", "worldwide", "pgdm", "galen", "claude",
    "office space", "commercial property", "science project", "television advertisement",
    "remote sensing", "startup company", "healthtech startups", "galen growth startups",
    "top healthtech startups", "healthtech nordic", "vacant", "lease", "transformation",
    "hardest sport", "best sport", "high school wrestling", "remember wrestling",
    "career because", "emdr", "aspect warriors", "phoenix lords", "long war",
}


def _final_clean(tags: list[str]) -> list[str]:
    import re

    out: list[str] = []
    for t in tags:
        n = normalize_tag(t)
        if not n or n in _FINAL_JUNK:
            continue
        toks = n.replace(".", " ").split()
        if len(toks) >= 4:
            continue
        # Drop dialogue crumbs (word-boundary, not substring — "had" ∉ "manhattan")
        if re.search(r"\b(couldn|wouldn|had|she|him|them|very|love|legends|bomb|squad)\b", n):
            continue
        if "'" in n:
            continue
        if n.endswith((" by", " too", " out", " in", " of", " for", " the", " most")):
            continue
        if n.startswith(("the ", "very ", "love ", "tech in ", "generation ")):
            continue
        out.append(n)
    return filter_validate_safe(out)


def select_wiki_ground(
    task,
    llm_tags: list[str] | None = None,
    *,
    budget: TimeBudget | None = None,
) -> list[str]:
    """Run WikiGround tag pipeline for a conversation (or survey) task."""
    texts = extract_task_texts(task)
    primary = texts["primary"] or ""
    enrichment = texts["enrichment"] or ""
    combined = texts["combined"] or ""
    if not combined.strip():
        return filter_validate_safe(llm_tags or [])[:10]

    policy = get_tag_policy(getattr(task, "type", "") or "conversation_tagging")
    min_tags = policy.min_tags
    target = policy.target_tags
    max_tags = policy.max_tags

    # Optional override for faster/slower linking (keep low — Wikidata rate-limits)
    mention_limit = int(os.environ.get("WIKI_MENTION_LIMIT", "28"))
    link_workers = int(os.environ.get("WIKI_LINK_WORKERS", "4"))

    # 1) Segment
    segments = segment_dialogue(primary, enrichment)
    if budget and budget.expired():
        return filter_validate_safe(llm_tags or [])[:max_tags]

    # 2) Mentions — prefer enrichment-heavy / high-score seeds for linking budget
    mentions = extract_mentions(segments, primary, enrichment, limit=mention_limit)
    if budget and budget.remaining() < 1.5:
        # Skip network linking — surface mentions only
        surface = [m.text for m in mentions]
        return filter_validate_safe(surface + list(llm_tags or []))[:max_tags]

    # Link only the strongest seeds (search API is cached; keep fan-out small)
    enr_l = enrichment.lower()
    # Shrink link budget when time is tight
    link_cap = min(12, mention_limit)
    if budget is not None:
        rem = budget.remaining()
        if rem < 3.0:
            link_cap = 0
        elif rem < 5.0:
            link_cap = min(link_cap, 4)
        elif rem < 7.0:
            link_cap = min(link_cap, 8)

    # Prefer enrichment / geo seeds — ReadyAI GT is headline-heavy; skips dialogue chatter
    def _seed_key(m):
        in_enr = 3.0 if enr_l and (m.text in enr_l or evidenced_in_enr(m.text, enr_l)) else 0.0
        src = 2.0 if m.source in {"geo", "anchor", "ner"} else 0.5
        return (in_enr, src, m.score)

    def evidenced_in_enr(tag: str, text: str) -> bool:
        from custom_miner.tag_pipeline.validate_safe import evidenced_in

        return evidenced_in(tag, text)

    ranked_seeds = sorted(mentions, key=_seed_key, reverse=True)
    # Keep at most 2 primary-only seeds for recall
    link_seeds = []
    primary_only = 0
    for m in ranked_seeds:
        in_enr = enr_l and evidenced_in_enr(m.text, enr_l)
        if not in_enr and m.source not in {"geo"}:
            if primary_only >= 2:
                continue
            primary_only += 1
        link_seeds.append(m)
        if len(link_seeds) >= link_cap:
            break

    # 3) Link (wbsearchentities only by default — descriptions drive expansion)
    if link_cap == 0:
        linked = []
    else:
        linked = link_mentions(
            link_seeds,
            context=combined[:6000],
            max_workers=min(link_workers, 4),
            limit=link_cap,
        )
    linked_mentions = {e.mention.lower() for e in linked}
    unlinked = [m.text for m in mentions if m.text.lower() not in linked_mentions]

    if budget and budget.expired():
        return filter_validate_safe([e.label for e in linked] + unlinked)[:max_tags]

    # 4) Expand via KG
    expanded = expand_entities(linked)

    # 5) Evidence gate
    gated = gate_candidates(
        expanded,
        primary=primary,
        enrichment=enrichment,
        unlinked_mentions=unlinked[:20],
        llm_tags=list(llm_tags or [])[:14],
    )

    # 6) Select
    selected = select_tags(
        gated,
        primary=primary,
        enrichment=enrichment,
        min_tags=min_tags,
        target_tags=target,
        max_tags=max_tags,
    )

    if len(selected) < 3:
        fallback = filter_validate_safe(
            [m.text for m in mentions] + list(llm_tags or [])
        )
        selected = filter_validate_safe(selected + fallback)[:max_tags]

    selected = _final_clean(selected)
    if len(selected) < 3:
        selected = _final_clean(
            selected + [m.text for m in mentions] + list(llm_tags or [])
        )

    bt.logging.info(
        f"wiki_ground segs={len(segments)} mentions={len(mentions)} "
        f"linked={len(linked)} expanded={len(expanded)} gated={len(gated)} "
        f"selected={len(selected)}"
    )
    return selected[:max_tags]
