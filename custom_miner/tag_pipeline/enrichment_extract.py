"""Enrichment-first entity harvest: titles, domains, codecs, platforms."""

from __future__ import annotations

import re

from custom_miner.tag_pipeline.anchors import clean_text
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import normalize_tag

_RE_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:news|gov|io|ai|com|org)\b", re.I)
_RE_MEDIA = re.compile(
    r"\b("
    r"avif|heif|mpeg|av1\s+codec|mp4\s+container|heif\s+container|"
    r"iso\s+base\s+media\s+file\s+format|isobmff|"
    r"joe\s+rogan|the\s+joe\s+rogan\s+experience|mike\s+tyson|"
    r"hugging\s+face|google\s+cloud|crunchbase|semrush|startupblink|"
    r"mantis\s+vc|lavote\.gov|hardwire\.news|cosmicbook\.news|"
    r"registrar[- ]recorder(?:/county\s+clerk)?"
    r")\b",
    re.I,
)
_RE_TITLE = re.compile(r"^[A-Z][^.\n|]{8,80}")


def extract_enrichment_entities(enrichment: str, *, limit: int = 16) -> list[str]:
    """High-priority tags from enrichment titles/snippets/domains."""
    if not enrichment:
        return []
    cleaned = clean_text(enrichment)
    blob = cleaned or enrichment
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        n = normalize_tag(raw)
        if not n or n in seen:
            return
        if n not in filter_validate_safe([n]):
            return
        seen.add(n)
        out.append(n)

    # Case-insensitive domain hunt (LAVote.gov etc.)
    for m in re.finditer(r"\b(?:[a-z0-9-]+\.)+(?:news|gov|io|ai|com|org)\b", blob, re.I):
        _add(m.group(0).lower())
    for m in _RE_MEDIA.finditer(blob):
        _add(m.group(1))
    # Explicit gov mentions without dotted form
    if "lavote" in blob.lower():
        _add("lavote.gov")
    # First lines often carry article titles
    for line in blob.splitlines()[:12]:
        line = line.strip()
        if not line or len(line) < 8:
            continue
        if _RE_TITLE.match(line) or line[:1].isupper():
            # Take leading proper chunk up to dash/pipe
            head = re.split(r"\s+[·|–—-]\s+|\s{2,}", line, maxsplit=1)[0].strip()
            if 3 <= len(head.split()) <= 6:
                _add(head)
            # Also bigrams of title tokens for people
            toks = re.findall(r"[A-Z][a-z]+", head)
            if len(toks) >= 2:
                _add(f"{toks[0]} {toks[1]}")
        if len(out) >= limit:
            break

    return out[:limit]
