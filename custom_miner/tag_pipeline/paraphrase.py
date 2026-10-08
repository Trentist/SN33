"""Semantic near-miss paraphrases — structural / type aliases only."""

from __future__ import annotations

from custom_miner.tag_pipeline.canonical import expand_canonical
from custom_miner.tag_pipeline.validate_safe import near_dupe
from custom_miner.tag_pipeline.validate_safe import normalize_tag


def domain_uniques_from_text(text: str, *, limit: int = 10) -> list[str]:
    """Unused legacy hook — keep for imports; prefer expand_canonical."""
    del text, limit
    return []


def expand_paraphrases(tags: list[str], *, max_per_tag: int = 2, limit: int = 24) -> list[str]:
    """Expand via general structural canonical rules (not problem answer keys)."""
    expanded = expand_canonical(tags, max_per_seed=max_per_tag, limit=limit)
    seeds = [normalize_tag(t) for t in tags if normalize_tag(t)]
    out: list[str] = []
    for cand in expanded:
        if any(near_dupe(cand, s, thr=0.45) for s in seeds + out):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out
