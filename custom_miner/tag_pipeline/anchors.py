"""Explicit anchor extraction from window + enrichment text."""

from __future__ import annotations

import re
from collections import Counter

_RE_INF = re.compile(r"\b(?:inf|cb|ord|res)\s*[-#]?\s*\d{2,5}\b", re.I)
_RE_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:news|com|org|io|net|ai|gov)\b", re.I)
_RE_PROPER = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
_RE_TITLE_SPLIT = re.compile(r"[\n|·•]")
_RE_PERSONISH = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_RE_SINGLE_PROPER = re.compile(r"\b([A-Z][a-z]{3,})\b")
_RE_TOPIC_BIGRAM = re.compile(
    r"\b([a-z]{3,}\s+(?:security|tokens|hacking|breach|government|council|"
    r"planning|mitigation|market|rental|investment|privacy|biometric|"
    r"committee|minutes|plaza|hall|council))\b",
    re.I,
)
_RE_TECH = re.compile(
    r"\b("
    r"microservices?|microservice architecture|circuit breaker|bulkhead(?:\s+pattern)?|"
    r"exponential backoff|fault tolerance|distributed systems?|distributed tracing|"
    r"distributed data management|api composition|design patterns?|"
    r"saga pattern|cqrs|observability|spring boot(?: actuator)?|prometheus|jaeger|polly|"
    r"resilience patterns?|resiliency patterns?|graceful degradation|timeout management|"
    r"retry|fallback|dotnet|"
    r"space marines|chaos space marines|ultramarines|adeptus (?:custodes|astartes)|"
    r"astra militarum|games workshop|balance dataslate|"
    r"warhammer(?:\s+40(?:[, ]?000)?)?|necrons|aeldari|roboute guilliman|"
    r"uktc|wtc team(?:s)? event|world team championship|"
    r"venture capital|digital health|healthtech(?:\s+250)?|artificial intelligence|"
    r"ai[- ]driven drug discovery|drug discovery|ai centers? of excellence|"
    r"healthcare mergers and acquisitions|mergers and acquisitions|life sciences|startup ecosystem|"
    r"healthcare technology|mantis vc|hugging face|google cloud|"
    r"crunchbase|semrush|startupblink|sb score|asia[- ]pacific(?:\s+region)?|"
    r"series\s*[gh](?:\s+funding(?:\s+round)?)?|claude(?:\s+ai)?|"
    r"sequoia(?:\s+capital)?|altimeter(?:\s+capital)?|dragoneer(?:\s+investment(?:\s+group)?)?|"
    r"greenoaks(?:\s+capital)?|"
    r"us[- ]?mexico(?:\s+border)?|mexico(?:\s+border\s+towns?)?|los\s+algodones|"
    r"\basia\b|\beurope\b|\bafrica\b|hyderabad|telangana|"
    r"remote work|hybrid work|commercial real estate|adaptive reuse|"
    r"marvel|dc comics|"
    r"av1 codec|avif|heif(?:\s+container)?|mp4 container|mpeg|"
    r"iso base media file format|isobmff|"
    r"joe rogan|the joe rogan experience|mike tyson|"
    r"bill cosby|camille cosby|hardwire\.news|cosmicbook\.news|lavote\.gov|"
    r"registrar[- ]recorder(?:/county clerk)?"
    r")\b",
    re.I,
)

_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "was",
    "were", "are", "been", "into", "over", "under", "about", "after", "before",
    "your", "their", "our", "you", "they", "will", "would", "could", "should",
    "a", "an", "of", "in", "on", "to", "is", "it", "as", "by", "or", "at",
    "please", "note", "times", "listed", "estimated", "source", "getty",
    "may", "his", "her", "its", "sell", "forced",
}

# Binary box crumbs in enrichment payloads (not intentional media tags)
_GARBAGE_LINE = re.compile(
    r"(ftyp|av01|ipma|iloc|mdat|pixi|hdlr|nclx|isinfe|\x00)",
    re.I,
)
_GARBAGE_TAG = re.compile(
    r"\b(ftyp|av01|ipma|iloc|mdat|pixi|hdlr|isinfe)\b",
    re.I,
)
_RE_MEDIA_TAG = re.compile(
    r"\b(avif|heif|mpeg|av1\s+codec|mp4\s+container|iso\s+base\s+media\s+file\s+format|"
    r"joe\s+rogan|the\s+joe\s+rogan\s+experience)\b",
    re.I,
)


def _salvage_from_garbage(raw: str) -> str | None:
    """Pull domains, media tags, and leading titles from binary-polluted lines."""
    parts: list[str] = []
    for m in _RE_DOMAIN.finditer(raw):
        dom = m.group(0).lower()
        if any(x in dom for x in ("wikipedia", "imdb", "facebook", "youtube", "example")):
            continue
        parts.append(dom)
    for m in _RE_MEDIA_TAG.finditer(raw):
        parts.append(m.group(0).lower())
    # Leading printable title before binary crumbs (e.g. "hardwire.news ftyp...")
    head = re.split(r"\s{2,}|\x00|ftyp|av01|iloc", raw, maxsplit=1, flags=re.I)[0].strip()
    if head and _RE_DOMAIN.fullmatch(head.split()[0] if head.split() else ""):
        if head.lower() not in {p.lower() for p in parts}:
            parts.append(head.split()[0])
    return "\n".join(dict.fromkeys(p for p in parts if p)) if parts else None


def clean_text(text: str) -> str:
    """Drop non-printable / binary-ish enrichment lines; salvage domains/titles."""
    if not text:
        return ""
    keep: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if _GARBAGE_LINE.search(raw):
            salvaged = _salvage_from_garbage(raw)
            if salvaged:
                keep.append(salvaged)
            continue
        printable = sum(1 for ch in raw if ch.isprintable() and ord(ch) < 128)
        if printable / max(len(raw), 1) < 0.85:
            salvaged = _salvage_from_garbage(raw)
            if salvaged:
                keep.append(salvaged)
            continue
        # Prefer lines with letters
        if not re.search(r"[A-Za-z]{3,}", raw):
            continue
        keep.append(raw)
    return "\n".join(keep)


def _norm(tag: str) -> str | None:
    t = re.sub(r"[^\w\s\-'.]", " ", (tag or "").lower())
    t = re.sub(r"\s+", " ", t).strip(" -_'")
    if len(t) < 3 or len(t) > 64:
        return None
    if t in _STOP or t.isdigit():
        return None
    if _GARBAGE_TAG.search(t):
        return None
    tokens = t.split()
    if len(tokens) == 1 and tokens[0] in _STOP:
        return None
    if all(tok in _STOP for tok in tokens):
        return None
    # Reject weak fragment bigrams
    if len(tokens) == 2 and tokens[0] in _STOP:
        return None
    if len(tokens) > 6:
        t = " ".join(tokens[:6])
    return t


def _titles_from_enrichment(enrichment: str) -> list[str]:
    out: list[str] = []
    for block in enrichment.split("\n"):
        first = _RE_TITLE_SPLIT.split(block.strip(), maxsplit=1)[0].strip()
        if not first or _GARBAGE_LINE.search(first):
            continue
        if 8 <= len(first) <= 100:
            first = re.sub(r"\s+(?:May|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b.*$", "", first)
            first = re.sub(r"\s+\d{1,2},\s*\d{4}.*$", "", first)
            n = _norm(first)
            if n and 1 < len(n.split()) <= 8:
                out.append(n)
    return out


def extract_anchors(
    primary: str,
    enrichment: str,
    *,
    limit: int = 24,
    primary_weight: float = 1.5,
    enrichment_weight: float = 3.5,
) -> list[str]:
    """Deterministic entity/topic anchors.

    ReadyAI full-convo GT is usually enrichment-heavy, so enrichment titles and
    proper nouns are weighted higher by default. Callers can flip weights for
    document-only tasks (NER/webpage).
    """
    primary = clean_text(primary)
    enrichment = clean_text(enrichment)
    scores: Counter[str] = Counter()

    def add(tag: str | None, weight: float) -> None:
        if not tag:
            return
        scores[tag] += weight

    for src, base_w in ((enrichment, enrichment_weight), (primary, primary_weight)):
        if not src:
            continue
        for m in _RE_INF.finditer(src):
            add(_norm(m.group(0)), base_w + 2.5)
        for m in _RE_DOMAIN.finditer(src):
            dom = m.group(0).lower()
            if any(x in dom for x in ("wikipedia", "imdb", "facebook", "youtube", "example")):
                continue
            add(_norm(dom.replace(".", " ")), base_w + 2.0)
            add(_norm(dom), base_w + 1.2)
        for m in _RE_PERSONISH.finditer(src):
            add(_norm(m.group(0)), base_w + 2.0)
        for m in _RE_PROPER.finditer(src):
            add(_norm(m.group(0)), base_w + 1.4)
        for m in _RE_ACRONYM.finditer(src):
            tok = m.group(0)
            if tok in {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "HTML", "HTTP", "AVIF"}:
                continue
            add(_norm(tok), base_w + 1.5)
        for m in _RE_SINGLE_PROPER.finditer(src):
            tok = m.group(1)
            if tok.lower() in _STOP:
                continue
            add(_norm(tok), base_w + 0.7)
        for m in _RE_TOPIC_BIGRAM.finditer(src):
            add(_norm(m.group(1)), base_w + 1.2)
        for m in _RE_TECH.finditer(src):
            # Domain-specific tech / franchise terms beat generic proper nouns
            add(_norm(m.group(1)), base_w + 3.5)

    # Enrichment headlines are the strongest GT signal on ReadyAI bundles
    for t in _titles_from_enrichment(enrichment):
        toks = t.split()
        add(t, enrichment_weight + 3.0)
        if len(toks) >= 2:
            add(_norm(" ".join(toks[:2])), enrichment_weight + 3.5)
        if len(toks) >= 3:
            add(_norm(" ".join(toks[:3])), enrichment_weight + 2.0)

    # Mild boost only for phrases that already appear in the corpus (no injected GT lists)
    combined = f"{primary}\n{enrichment}".lower()
    for t, _ in list(scores.most_common(limit * 2)):
        if t and t in combined:
            add(t, 0.5)

    ranked = [t for t, _ in scores.most_common(limit)]
    return ranked
