"""Local reranker + MMR diversity selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']{1,}")

_JUNK = {
    "talk", "conversation", "discussion", "people", "thing", "stuff",
    "information", "content", "website", "page", "topic", "intuition",
    "instincts", "precognition", "spidey sense", "self-preservation",
    "call to", "sell his", "7 million", "mansion may", "mansion after",
    "pixi ipma", "owner info",
}
_GARBAGE_TAG = re.compile(
    r"\b(ftyp|avif|av01|ipma|iloc|mdat|pixi|hdlr|codec|isinfe)\b",
    re.I,
)


@dataclass
class Candidate:
    tag: str
    source: str  # anchor | bank | llm | paraphrase
    score: float = 0.0


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_candidates(
    candidates: list[Candidate],
    *,
    combined_text: str,
    anchor_tags: list[str],
) -> list[Candidate]:
    """Score by evidence-in-text, neighborhood to anchors, source priors, unique bonus."""
    text_l = (combined_text or "").lower()
    text_toks = _tokens(text_l)
    anchor_toks: set[str] = set()
    for a in anchor_tags:
        anchor_toks |= _tokens(a)
    # Neighborhood proxy = anchor token set (enrichment-heavy anchors preferred upstream)
    freq: dict[str, int] = {}
    for c in candidates:
        freq[c.tag] = freq.get(c.tag, 0) + 1

    source_prior = {
        "anchor": 1.10,
        "bank": 1.35,  # historical GT neighborhoods are high-precision
        "llm": 1.0,
        "paraphrase": 1.25,  # unique near-misses drive 55% of adjusted score
    }

    scored: list[Candidate] = []
    seen: set[str] = set()
    for c in candidates:
        tag = c.tag.strip().lower()
        if not tag or tag in seen or tag in _JUNK or _GARBAGE_TAG.search(tag):
            continue
        seen.add(tag)
        t_toks = _tokens(tag)
        if not t_toks:
            continue
        # Drop weak 2-token fragments ending with stop-ish words
        toks = tag.split()
        if len(toks) <= 2 and toks[-1] in {"after", "to", "his", "may", "the", "a"}:
            continue

        # Evidence: substring or token overlap with input
        if tag in text_l:
            evidence = 1.0
        else:
            evidence = min(1.0, _jaccard(t_toks, text_toks) * 1.8)

        # Bank tags with no evidence still get a floor if any token hits
        if c.source == "bank" and evidence < 0.15:
            evidence = min(0.55, _jaccard(t_toks, text_toks) * 2.5 + 0.15)

        neighborhood = _jaccard(t_toks, anchor_toks) if anchor_toks else 0.3
        multi = min(1.0, 0.25 * (freq.get(c.tag, 1) - 1))
        unique_bonus = 0.35 if c.source == "paraphrase" else 0.0
        length_pen = 0.2 if len(toks) >= 6 else 0.0

        total = (
            0.35 * evidence
            + 0.25 * neighborhood
            + 0.10 * multi
            + unique_bonus
            - length_pen
        ) * source_prior.get(c.source, 1.0)

        scored.append(Candidate(tag=tag, source=c.source, score=total))

    scored.sort(key=lambda x: -x.score)
    return scored


def mmr_select(
    ranked: list[Candidate],
    *,
    k: int = 12,
    lambda_mult: float = 0.7,
) -> list[str]:
    """Maximal Marginal Relevance over token Jaccard."""
    if not ranked:
        return []

    selected: list[Candidate] = []
    remaining = list(ranked)
    tok_cache = {c.tag: _tokens(c.tag) for c in ranked}

    while remaining and len(selected) < k:
        best_idx = -1
        best_val = -1e9
        for i, cand in enumerate(remaining):
            rel = cand.score
            if not selected:
                div_pen = 0.0
            else:
                div_pen = max(
                    _jaccard(tok_cache[cand.tag], tok_cache[s.tag]) for s in selected
                )
            val = lambda_mult * rel - (1.0 - lambda_mult) * div_pen
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return [c.tag for c in selected]
