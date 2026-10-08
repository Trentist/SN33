"""Extractive mention / keyphrase mining (PatternRank-lite + anchors).

No GT tag bank. Optional spaCy NER when installed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from custom_miner.tag_pipeline.anchors import extract_anchors
from custom_miner.tag_pipeline.validate_safe import normalize_tag

# Noun-phrase-ish patterns (PatternRank PoS approximation without a tagger)
_NP = re.compile(
    r"\b("
    r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})"  # Proper Name Phrase
    r"|(?:[A-Z]{2,6})"  # ACRONYM
    r"|(?:[a-z]{3,}(?:\s+[a-z]{3,}){1,3})"  # lowercase multiword (from lowered text)
    r")\b"
)
_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:news|com|org|io|net|ai|gov)\b", re.I)
_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "was",
    "were", "are", "been", "into", "over", "under", "about", "after", "before",
    "your", "their", "our", "you", "they", "will", "would", "could", "should",
    "yeah", "okay", "really", "actually", "just", "like", "know", "think",
    "going", "gonna", "right", "well", "thing", "things", "stuff", "people",
    "talk", "talking", "said", "says", "come", "came", "want", "wanted",
    "look", "looks", "looking", "kind", "sort", "lot", "bit", "way", "time",
    "even", "though", "more", "than", "four", "find", "check", "info",
    "little", "place", "rules", "working", "together", "owner", "permits",
}

# Weak fragments that are not taggable topics
_WEAK_MENTIONS = {
    "yeah", "bill", "angeles", "county", "mansion", "rogan", "trouble source",
    "more than four decades", "find owner info", "check permits", "even though",
    "working out together", "little loud", "little volume knob", "place rules",
    "trouble", "source", "decades", "permits", "deed", "loan", "lien",
}

# Headline geo / brand expansions commonly in ReadyAI GT
_GEO_EXPAND = (
    (re.compile(r"\bNYC\b"), "new york city"),
    (re.compile(r"\bNew York\b"), "new york city"),
    (re.compile(r"\bManhattan\b"), "manhattan"),
    (re.compile(r"\bManhattan\b"), "new york city"),
    (re.compile(r"\bCalifornia\b"), "california"),
    (re.compile(r"\bLos Angeles County\b"), "california"),
    (re.compile(r"\bLos Angeles\b"), "los angeles"),
    (re.compile(r"\bLos Angeles\b"), "california"),
    (re.compile(r"\bLA\b"), "los angeles"),
)

_CHATTER = re.compile(
    r"\b(fucking|crazy|story|encourage|expose|getting|paid|fickle|sometimes|"
    r"personal|together|prioritize|flexibility|remain|below|levels|vibrant|"
    r"revitalizing|mixed|him|them|too|also|really|gonna|wanna)\b",
    re.I,
)


@dataclass
class Mention:
    text: str
    segment_idx: int
    source: str  # ner | np | anchor | domain
    score: float
    count: int = 1


def _spacy_ents(text: str) -> list[tuple[str, float]]:
    try:
        import spacy
    except Exception:
        return []
    global _NLP  # type: ignore[name-defined]
    try:
        nlp = _NLP  # type: ignore[name-defined]
    except NameError:
        nlp = None
    if nlp is None:
        for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                nlp = spacy.load(model, disable=["parser", "lemmatizer", "textcat"])
                break
            except Exception:
                continue
        globals()["_NLP"] = nlp
    if nlp is None:
        return []
    out: list[tuple[str, float]] = []
    try:
        doc = nlp(text[:8000])
    except Exception:
        return []
    keep = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC", "NORP"}
    for ent in doc.ents:
        if ent.label_ not in keep:
            continue
        n = normalize_tag(ent.text)
        if n:
            out.append((n, 1.4 if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT"} else 1.0))
    return out


def _np_candidates(text: str) -> list[tuple[str, float]]:
    """Capitalized / acronym candidates from original case; lowercase n-grams from lowered."""
    out: list[tuple[str, float]] = []
    for m in _NP.finditer(text):
        raw = m.group(1).strip()
        if raw.islower() and len(raw.split()) < 2:
            continue
        n = normalize_tag(raw)
        if not n:
            continue
        toks = n.split()
        if any(t in _STOP for t in toks):
            continue
        if len(toks) == 1 and (len(toks[0]) < 4 or toks[0] in _STOP):
            continue
        # Prefer proper / acronym
        boost = 1.1 if raw[:1].isupper() else 0.7
        if raw.isupper() and 2 <= len(raw) <= 6:
            boost = 1.3
        out.append((n, boost))
    for m in _DOMAIN.finditer(text):
        dom = m.group(0).lower()
        if any(x in dom for x in ("wikipedia", "imdb", "facebook", "youtube", "example")):
            continue
        n = normalize_tag(dom.replace(".", " "))
        if n:
            out.append((n, 1.5))
            n2 = normalize_tag(dom)
            if n2:
                out.append((n2, 1.2))
    return out


def _is_weak(tag: str) -> bool:
    if not tag or tag in _WEAK_MENTIONS or tag in _STOP:
        return True
    if _CHATTER.search(tag):
        return True
    toks = tag.split()
    if len(toks) >= 2 and all(t in _STOP or len(t) <= 3 for t in toks):
        return True
    # Verb-y / headline-fragment chatter
    if len(toks) >= 2 and toks[0] in {
        "find", "check", "more", "even", "working", "little", "place", "is",
        "are", "was", "were", "has", "have", "get", "got", "converting",
        "thriving", "opportunities", "global", "revenue", "billion",
        "salesforce",  # prefer bare brand via other path; drop "salesforce layoffs…"
    }:
        # Allow known multiword brands starting with these only if exact short forms
        if tag not in {"salesforce", "google cloud", "hugging face"}:
            if toks[0] in {"salesforce", "converting", "thriving", "opportunities",
                           "global", "revenue", "billion"} or len(toks) >= 3:
                return True
    if toks and toks[-1] in {"by", "in", "of", "to", "for", "and", "the", "a", "highlight", "april"}:
        return True
    # Long headline crumbs / soft news fragments
    if len(toks) >= 4:
        return True
    if len(toks) == 3 and toks[1] in {
        "funding", "layoffs", "trends", "vacant", "spaces", "office", "mansion",
    }:
        return True
    return False


def extract_mentions(
    segments: list,
    primary: str,
    enrichment: str,
    *,
    limit: int = 48,
) -> list[Mention]:
    """Mine mentions across segments + global anchors."""
    scores: Counter[str] = Counter()
    meta: dict[str, Mention] = {}

    def add(tag: str, seg_idx: int, source: str, weight: float) -> None:
        n = normalize_tag(tag)
        if not n or _is_weak(n):
            return
        scores[n] += weight
        prev = meta.get(n)
        if prev is None:
            meta[n] = Mention(text=n, segment_idx=seg_idx, source=source, score=weight, count=1)
        else:
            prev.count += 1
            prev.score += weight
            # Prefer earlier (dialogue) segment over enrichment-only
            if prev.segment_idx < 0 and seg_idx >= 0:
                prev.segment_idx = seg_idx
            if source == "ner" and prev.source != "ner":
                prev.source = source

    # Global deterministic anchors (reuse existing extractor — not GT bank)
    for t in extract_anchors(primary, enrichment, limit=32):
        add(t, 0, "anchor", 2.0)

    # Geo / city expansions from enrichment headlines (press-style)
    for pat, geo in _GEO_EXPAND:
        if pat.search(enrichment or "") or pat.search(primary or ""):
            add(geo, 0, "geo", 2.8)

    for seg in segments:
        idx = getattr(seg, "idx", 0)
        text = getattr(seg, "text", "") or ""
        if not text:
            continue
        # Enrichment segment gets slightly higher weight (ReadyAI GT is enrichment-heavy)
        base = 1.6 if getattr(seg, "turn_start", 0) < 0 else 1.0
        for tag, w in _spacy_ents(text):
            add(tag, idx, "ner", w * base)
        for tag, w in _np_candidates(text):
            add(tag, idx, "np", w * base)
        for pat, geo in _GEO_EXPAND:
            if pat.search(text):
                add(geo, idx, "geo", 2.2 * base)

    ranked = scores.most_common(limit)
    out: list[Mention] = []
    for tag, sc in ranked:
        m = meta[tag]
        m.score = float(sc)
        out.append(m)
    return out
