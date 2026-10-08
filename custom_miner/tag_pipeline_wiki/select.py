"""Salience ranking + MMR diversity selection."""

from __future__ import annotations

from collections import defaultdict

from custom_miner.tag_pipeline.validate_safe import dedupe_near
from custom_miner.tag_pipeline.validate_safe import evidenced_in
from custom_miner.tag_pipeline.validate_safe import near_dupe
from custom_miner.tag_pipeline_wiki.expand import ExpandedTag

_DEDUPE = 0.42


def _type_bucket(e: ExpandedTag) -> str:
    via = e.via_prop or ""
    if via in {"P17", "P131", "P27", "P495"}:
        return "geo"
    if via in {"P452", "P136", "P101", "P279", "P31"}:
        return "category"
    if via in {"", "mention", "alias", "surface", "llm"}:
        return "entity"
    return "other"


def select_tags(
    candidates: list[ExpandedTag],
    *,
    primary: str,
    enrichment: str,
    min_tags: int = 10,
    target_tags: int = 12,
    max_tags: int = 14,
) -> list[str]:
    """Rank by salience and pick a diverse set of 10–14 tags."""
    if not candidates:
        return []

    conv_l = (primary or "").lower()
    enr_l = (enrichment or "").lower()
    combined = (conv_l + "\n" + enr_l).strip()

    by_tag: dict[str, ExpandedTag] = {}
    weight_sum: dict[str, float] = defaultdict(float)
    for e in candidates:
        weight_sum[e.tag] += e.weight
        prev = by_tag.get(e.tag)
        if prev is None or e.weight > prev.weight:
            by_tag[e.tag] = e

    scored: list[tuple[float, ExpandedTag]] = []
    for tag, e in by_tag.items():
        if enr_l and evidenced_in(tag, enr_l):
            textual = 1.15  # ReadyAI GT is enrichment-heavy
        elif conv_l and evidenced_in(tag, conv_l):
            textual = 0.35  # dialogue name-drops are weaker than headlines
        else:
            textual = 0.0
        link_bonus = 0.12 if e.linked else 0.0
        via_bonus = 0.15 if e.via_prop in ("", "mention", "surface", "geo") else 0.0
        if e.via_prop in {"desc", "P452", "P136", "P131", "P17"}:
            via_bonus = 0.20  # press-style geo/category from WD descriptions
        # Penalize primary-only chatter when enrichment exists
        primary_only_pen = 0.0
        if enr_l and conv_l and evidenced_in(tag, conv_l) and not evidenced_in(tag, enr_l):
            if e.via_prop in ("", "mention", "surface", "np") and not e.linked:
                primary_only_pen = 0.45
        salience = (
            0.40 * weight_sum[tag]
            + 0.40 * textual
            + link_bonus
            + via_bonus
            - primary_only_pen
        )
        scored.append((salience, e))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[ExpandedTag] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    bucket_caps = {"entity": max_tags, "geo": 5, "category": 5, "other": 4}

    for _sal, e in scored:
        if len(selected) >= max_tags:
            break
        b = _type_bucket(e)
        if bucket_counts[b] >= bucket_caps.get(b, 4):
            continue
        if any(near_dupe(e.tag, s.tag, thr=_DEDUPE) for s in selected):
            continue
        # Front-load evidenced entities; demote random dialogue name-drops
        if len(selected) < 5 and not (
            evidenced_in(e.tag, combined) or e.via_prop in ("", "mention", "surface", "geo", "desc")
        ):
            continue
        # Require text evidence OR KG desc/geo expansion from an evidenced seed
        if not evidenced_in(e.tag, combined) and e.via_prop not in {"desc", "P17", "P131", "P452", "P136"}:
            if e.via_prop not in ("", "mention", "surface", "alias", "geo"):
                continue
            if not (evidenced_in(e.mention, combined) or evidenced_in(e.source_label.lower(), combined)):
                continue
        selected.append(e)
        bucket_counts[b] += 1

    if len(selected) < min_tags:
        have = {s.tag for s in selected}
        for _sal, e in scored:
            if e.tag in have:
                continue
            if any(near_dupe(e.tag, s.tag, thr=_DEDUPE) for s in selected):
                continue
            # Backfill only from enrichment / KG expansions — not dialogue chatter
            ok_backfill = (
                (enr_l and evidenced_in(e.tag, enr_l))
                or e.via_prop in {"desc", "geo", "P17", "P131", "P452", "P136", "alias"}
                or (e.linked and evidenced_in(e.mention, combined))
            )
            if not ok_backfill:
                continue
            selected.append(e)
            have.add(e.tag)
            if len(selected) >= target_tags:
                break

    tags = dedupe_near([e.tag for e in selected], thr=_DEDUPE)
    # Prefer quality over padding — allow down to 8 if we can't fill cleanly
    if len(tags) < 3:
        return tags
    return tags[:max_tags]
