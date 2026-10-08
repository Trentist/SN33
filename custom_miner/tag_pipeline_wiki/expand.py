"""1-hop Wikidata property expansion → candidate tag strings.

Replaces hand-built alias tables with live KG neighbors.
Also mines tags from Wikidata *descriptions* when claim fetch is rate-limited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from custom_miner.tag_pipeline_wiki.link import LinkedEntity
from custom_miner.tag_pipeline_wiki.link import fetch_entities
from custom_miner.tag_pipeline.validate_safe import normalize_tag

# Pull place / industry phrases from WD short descriptions (works without claims)
_DESC_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bcounty in ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.I), 0.85),
    (re.compile(r"\bborough of ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})", re.I), 0.9),
    (re.compile(r"\bcity in ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.I), 0.8),
    (re.compile(r"\bin ([A-Z][a-zA-Z]+),\s*United States", re.I), 0.75),
    (re.compile(r"\bmixed martial arts\b", re.I), 0.9),
    (re.compile(r"\bcomic books?\b", re.I), 0.85),
    (re.compile(r"\bsuperhero\b", re.I), 0.7),
    (re.compile(r"\bventure capital\b", re.I), 0.9),
    (re.compile(r"\bartificial intelligence\b", re.I), 0.9),
    (re.compile(r"\bdigital health\b", re.I), 0.85),
    (re.compile(r"\bhealthcare\b", re.I), 0.7),
]

# Property → expansion weight
_PROPS: dict[str, float] = {
    "P31": 0.75,   # instance of
    "P279": 0.70,  # subclass of
    "P361": 0.65,  # part of
    "P17": 0.80,   # country
    "P131": 0.85,  # located in admin territorial entity
    "P452": 0.90,  # industry
    "P136": 0.90,  # genre
    "P495": 0.55,  # country of origin
    "P106": 0.45,  # occupation
    "P27": 0.50,   # country of citizenship
    "P170": 0.40,  # creator
    "P123": 0.55,  # publisher
    "P127": 0.50,  # owned by
    "P749": 0.50,  # parent org
    "P178": 0.55,  # developer
    "P277": 0.40,  # programming language
    "P1056": 0.45, # product / material produced
    "P101": 0.50,  # field of work
}

# Over-generic WD class labels — never emit as tags
_GENERIC_LABELS = {
    "human", "male", "female", "person", "organism", "taxon",
    "organization", "organisation", "company", "business", "enterprise",
    "city", "town", "country", "sovereign state", "administrative territorial entity",
    "human settlement", "big city", "metropolis", "municipality",
    "wikimedia category", "wikimedia list article", "wikimedia disambiguation page",
    "literary work", "written work", "creative work", "work",
    "television series", "television program",  # keep more specific genres instead
    "website", "online newspaper", "periodical",
    "occurrence", "event", "recurring event",
    "academic discipline", "field of study",
}


@dataclass
class ExpandedTag:
    tag: str
    source_qid: str
    source_label: str
    via_prop: str  # "" for direct entity label
    weight: float
    linked: bool = True
    mention: str = ""


def _en_label(entity: dict) -> str:
    labels = entity.get("labels") or {}
    en = labels.get("en") or {}
    return (en.get("value") or "").strip()


def _en_aliases(entity: dict) -> list[str]:
    aliases = entity.get("aliases") or {}
    en = aliases.get("en") or []
    out = []
    for a in en[:8]:
        if isinstance(a, dict) and a.get("value"):
            out.append(str(a["value"]))
    return out


def _claim_qids(entity: dict, prop: str) -> list[str]:
    claims = (entity.get("claims") or {}).get(prop) or []
    qids: list[str] = []
    for c in claims:
        snak = (c or {}).get("mainsnak") or {}
        if snak.get("snaktype") != "value":
            continue
        dv = snak.get("datavalue") or {}
        val = dv.get("value")
        if isinstance(val, dict) and val.get("id"):
            qids.append(val["id"])
    return qids


def _ok_label(label: str) -> str | None:
    n = normalize_tag(label)
    if not n:
        return None
    if n in _GENERIC_LABELS:
        return None
    # Drop "X of Y" bureaucracy unless short
    if n.startswith("wikimedia") or "disambiguation" in n:
        return None
    if len(n.split()) > 5:
        return None
    return n


def _desc_tags(description: str) -> list[tuple[str, float]]:
    """Extract candidate tags from a Wikidata description string."""
    if not description:
        return []
    out: list[tuple[str, float]] = []
    for pat, w in _DESC_PATTERNS:
        m = pat.search(description)
        if not m:
            continue
        if m.lastindex:
            out.append((m.group(1), w))
        else:
            out.append((m.group(0), w))
    # "X in California, United States" style trailing state
    m = re.search(r",\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*United States", description)
    if m:
        out.append((m.group(1), 0.7))
    return out


def expand_entities(linked: list[LinkedEntity]) -> list[ExpandedTag]:
    """Emit entity labels + description tags; optionally 1-hop claims.

    Default skips wbgetentities (Wikidata rate-limits hard). Set
    WIKI_FETCH_CLAIMS=1 to enable live claim expansion when cache is warm.
    """
    import os

    if not linked:
        return []

    fetch_claims = os.environ.get("WIKI_FETCH_CLAIMS", "0") == "1"
    entities: dict = {}
    neighbors: dict = {}
    if fetch_claims:
        qids = [e.qid for e in linked]
        entities = fetch_entities(qids)
        neighbor_ids: set[str] = set()
        for _qid, ent in entities.items():
            for prop in _PROPS:
                for nqid in _claim_qids(ent, prop)[:3]:
                    neighbor_ids.add(nqid)
        neighbors = fetch_entities(sorted(neighbor_ids)) if neighbor_ids else {}

    out: list[ExpandedTag] = []
    seen: set[str] = set()

    def emit(tag: str, source_qid: str, source_label: str, via: str, weight: float, mention: str) -> None:
        n = _ok_label(tag)
        if not n or n in seen:
            return
        seen.add(n)
        out.append(
            ExpandedTag(
                tag=n,
                source_qid=source_qid,
                source_label=source_label,
                via_prop=via,
                weight=weight,
                linked=True,
                mention=mention,
            )
        )

    for ent_link in linked:
        ent = entities.get(ent_link.qid) or {}
        label = _en_label(ent) or ent_link.label
        # Prefer short surface / alias forms ReadyAI GT uses (ufc, marvel)
        mention_n = normalize_tag(ent_link.mention)
        label_n = normalize_tag(label)
        if mention_n and len(mention_n) <= len(label_n or mention_n):
            emit(mention_n, ent_link.qid, label, "mention", 1.15 + 0.3 * ent_link.score, ent_link.mention)
        emit(label, ent_link.qid, label, "", 1.0 + 0.3 * ent_link.score, ent_link.mention)
        # Useful aliases that look like GT tags (short)
        for alias in list(ent_link.aliases or [])[:4] + _en_aliases(ent)[:4]:
            a = _ok_label(alias)
            if a and a != label_n:
                boost = 0.35 if mention_n and a == mention_n else 0.0
                emit(a, ent_link.qid, label, "alias", 0.55 + 0.2 * ent_link.score + boost, ent_link.mention)

        # Description mining — works even when claims fetch is 429'd
        desc = ent_link.description or ""
        if not desc:
            dens = (ent.get("descriptions") or {}).get("en") or {}
            desc = dens.get("value") or ""
        for phrase, w in _desc_tags(desc):
            emit(phrase, ent_link.qid, label, "desc", w * (0.7 + 0.3 * ent_link.score), ent_link.mention)

        for prop, pweight in _PROPS.items():
            for nqid in _claim_qids(ent, prop)[:2]:
                nent = neighbors.get(nqid) or {}
                nlabel = _en_label(nent)
                if not nlabel:
                    continue
                emit(
                    nlabel,
                    ent_link.qid,
                    label,
                    prop,
                    pweight * (0.7 + 0.3 * ent_link.score),
                    ent_link.mention,
                )

    return out
