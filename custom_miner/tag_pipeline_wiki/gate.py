"""Evidence gate: keep tags grounded in conversation or 1-hop from linked mentions."""

from __future__ import annotations

from custom_miner.tag_pipeline.validate_safe import evidenced_in
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import normalize_tag
from custom_miner.tag_pipeline_wiki.expand import ExpandedTag

_GENERIC = {
    "discussion", "conversation", "people", "life", "information", "opinion",
    "article", "articles", "media", "news", "topic", "topics", "general",
    "random", "stuff", "things", "update", "updates", "content", "podcast",
    "american", "actor", "comedian", "company", "organization", "website",
    "mixed martial arts promotion", "promotion company",
    "property", "foreclosure", "million", "vacant", "lease", "transformation",
    "worldwide", "galen", "claude", "blood", "sons", "pgdm",
    "united states", "television actor", "charter county of california",
    "borough of new york city", "consolidated city-county",
}

_WEAK_SINGLE = {
    "property", "foreclosure", "million", "vacant", "lease", "blood", "sons",
    "galen", "claude", "india", "jake", "paul", "craig", "hard", "times",
}


def gate_candidates(
    expanded: list[ExpandedTag],
    *,
    primary: str,
    enrichment: str,
    unlinked_mentions: list[str] | None = None,
    llm_tags: list[str] | None = None,
) -> list[ExpandedTag]:
    """Filter to evidenced surface tags OR 1-hop KG expansions from evidenced entities."""
    conv_l = (primary or "").lower()
    enr_l = (enrichment or "").lower()
    combined = (conv_l + "\n" + enr_l).strip()

    # Mentions that appear in text are allowed to seed expansions
    evidenced_sources: set[str] = set()
    for e in expanded:
        seed = normalize_tag(e.mention) or ""
        label = normalize_tag(e.source_label) or ""
        if seed and (evidenced_in(seed, combined) or evidenced_in(seed, enr_l)):
            evidenced_sources.add(e.source_qid)
        if label and (evidenced_in(label, combined) or evidenced_in(label, enr_l)):
            evidenced_sources.add(e.source_qid)

    kept: list[ExpandedTag] = []
    seen: set[str] = set()

    def keep(e: ExpandedTag, *, boost: float = 0.0) -> None:
        n = normalize_tag(e.tag)
        if not n or n in seen or n in _GENERIC:
            return
        if n in _WEAK_SINGLE or (len(n.split()) == 1 and n in _WEAK_SINGLE):
            return
        if n not in filter_validate_safe([n]):
            return
        seen.add(n)
        e.tag = n
        e.weight = e.weight + boost
        kept.append(e)

    for e in expanded:
        n = normalize_tag(e.tag)
        if not n:
            continue
        in_text = evidenced_in(n, combined)
        # Direct labels / mentions must be in text (or very close)
        if e.via_prop in ("", "mention", "alias", "surface", "geo", "llm"):
            if in_text or evidenced_in(e.mention, combined) or e.via_prop == "geo":
                keep(e, boost=0.25 if enr_l and evidenced_in(n, enr_l) else 0.0)
            continue
        # desc / property expansions from evidenced linked entities
        if e.via_prop == "desc" or e.via_prop.startswith("P"):
            if e.source_qid in evidenced_sources or evidenced_in(e.mention, combined):
                keep(e, boost=0.10 if in_text else 0.0)
            continue
        if e.source_qid in evidenced_sources:
            keep(e, boost=0.10 if in_text else 0.0)

    # Unlinked but evidenced surface mentions (domains, obscure brands)
    for m in unlinked_mentions or []:
        n = normalize_tag(m)
        if not n or n in seen:
            continue
        if evidenced_in(n, combined):
            keep(
                ExpandedTag(
                    tag=n,
                    source_qid="",
                    source_label=n,
                    via_prop="surface",
                    weight=0.85,
                    linked=False,
                    mention=n,
                ),
                boost=0.15 if enr_l and evidenced_in(n, enr_l) else 0.0,
            )

    # LLM tags only if evidenced — never free-form invention
    for t in llm_tags or []:
        n = normalize_tag(t) if isinstance(t, str) else None
        if not n or n in seen:
            continue
        if evidenced_in(n, combined):
            keep(
                ExpandedTag(
                    tag=n,
                    source_qid="",
                    source_label=n,
                    via_prop="llm",
                    weight=0.70,
                    linked=False,
                    mention=n,
                )
            )

    return kept
