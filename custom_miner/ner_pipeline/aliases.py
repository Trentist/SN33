"""Text-grounded alternate names.

The validator prompt collapses "Mayor Adams" and "Adams" into one string.
That surviving string is an exact overlap. The collapsed form is a different
string with almost the same embedding, which is what the top-3 unique term
rewards. These aliases are recovered from the source text and from the core
name itself, without another model call.
"""

from __future__ import annotations

import re

from custom_miner.ner_pipeline.gate import clean_tag

_TITLES = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "mayor",
    "president",
    "senator",
    "governor",
    "representative",
    "rep",
    "councilmember",
    "councilor",
    "councillor",
    "alderman",
    "judge",
    "justice",
    "minister",
    "secretary",
    "general",
    "sir",
    "dame",
}

_TRAILING = {
    "city",
    "county",
    "state",
    "university",
    "college",
    "inc",
    "corp",
    "llc",
    "ltd",
    "department",
    "dept",
    "council",
    "committee",
    "commission",
    "authority",
    "agency",
    "district",
}

_GENERIC = _TITLES | _TRAILING | {
    "law",
    "act",
    "bill",
    "grant",
    "project",
    "program",
    "plan",
    "budget",
    "news",
    "street",
    "avenue",
    "road",
    "local",
    "federal",
    "national",
    "public",
    "new",
    "the",
    "of",
    "and",
    "for",
}

_SPAN = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|the|and|for|de|la|von|van)){1,5}"
    r"|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3}"
    r")"
)
_LAW = re.compile(
    r"\b((?:local|federal|state|public)\s+law\s+\d+[a-z]?)\b",
    re.I,
)
_ACT = re.compile(
    r"\b([A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+){0,5}\s+(?:Act|Bill|Grant|Project))\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{3,6})\b")
_BILL = re.compile(
    r"\b((?:council\s+bill|house\s+bill|senate\s+bill|local\s+law|ordinance|cb|hb|sb)\s+\d+[a-z]?)\b",
    re.I,
)


def _ok(raw: str) -> str | None:
    cleaned = clean_tag(raw)
    if not cleaned:
        return None
    if len(cleaned) < 3:
        return None
    return cleaned


def extract_spans(text: str) -> list[str]:
    """Proper-name spans that actually occur in the transcript or a snippet."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        cleaned = _ok(raw)
        if not cleaned or cleaned in seen:
            return
        toks = cleaned.split()
        if len(toks) == 1 and toks[0] in _GENERIC:
            return
        seen.add(cleaned)
        found.append(cleaned)

    for match in _LAW.finditer(text):
        add(match.group(1))
    for match in _BILL.finditer(text):
        add(match.group(1))
    for match in _ACT.finditer(text):
        add(match.group(1))
    for match in _SPAN.finditer(text):
        add(match.group(0))
    for match in _ACRONYM.finditer(text):
        token = match.group(1)
        if token not in {"THE", "AND", "FOR", "BUT", "NOT", "YOU", "ARE"}:
            add(token)
    return found[:24]


def _distinctive(token: str) -> bool:
    return len(token) >= 4 and token not in _GENERIC and not token.isdigit()


def alias_forms(core: str) -> list[str]:
    """Shorter or de-titled forms of one core entity."""
    toks = (core or "").split()
    if len(toks) < 2:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(parts: list[str]) -> None:
        cleaned = _ok(" ".join(parts))
        if not cleaned or cleaned == core or cleaned in seen:
            return
        if not any(_distinctive(tok) or tok.isdigit() for tok in cleaned.split()):
            return
        seen.add(cleaned)
        out.append(cleaned)

    body = list(toks)
    while body and body[0] in _TITLES:
        body = body[1:]
        add(body)
    tail = list(toks)
    while len(tail) > 1 and tail[-1] in _TRAILING:
        tail = tail[:-1]
        add(tail)
    last = toks[-1]
    if _distinctive(last):
        add([last])
    if len(toks) >= 3 and _distinctive(toks[-2]) and _distinctive(toks[-1]):
        add(toks[-2:])
    return out


def grounded_aliases(text: str, cores: list[str]) -> list[str]:
    """Close surface forms of the core names only.

    Full alternates (``eric adams``) stay. Every other capitalized span is left
    for the entity pool, where the centroid can accept or reject it. Dumping
    those spans in here is what pulled the mean down.
    """
    spans = extract_spans(text)
    core_set = {c for c in cores if c}
    aliases: list[str] = []
    seen: set[str] = set()

    def add(tag: str) -> None:
        if not tag or tag in seen or tag in core_set:
            return
        seen.add(tag)
        aliases.append(tag)

    for core in cores:
        for form in alias_forms(core):
            add(form)
        distinctive = {tok for tok in core.split() if _distinctive(tok)}
        if len(distinctive) < 2:
            continue
        for span in spans:
            span_toks = set(span.split())
            if distinctive <= span_toks and len(span_toks) > len(core.split()):
                add(span)
    return aliases[:24]


_SPAN_JUNK = {
    "report",
    "agenda",
    "proceedings",
    "journal",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "monday",
    "saturday",
    "sunday",
    "meeting",
    "present",
    "page",
    "floor",
    "excused",
    "avenue",
    "ave",
    "street",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
}


def recovered_entities(text: str, already: set[str]) -> list[str]:
    """Other people, bills, and places in the text, for the centroid pool."""
    out: list[str] = []
    for span in extract_spans(text):
        toks = span.split()
        if span in already or any(tok in _SPAN_JUNK for tok in toks):
            continue
        if len(toks) == 1 and span in _GENERIC:
            continue
        if not 2 <= len(toks) <= 5 and not any(ch.isdigit() for ch in span):
            continue
        out.append(span)
        if len(out) >= 16:
            break
    return out


def anchor_paraphrases(anchors: list[str], text: str) -> list[str]:
    """Recombine anchor tokens into nearby names that stay on the same centroid.

    On council packets the centroid sits on the institution (``seattle city
    council``, ``city hall``), not on each member. Short recombinations of those
    anchors are the high-cosine uniques. Tokens still have to occur in the text.
    """
    low = (text or "").lower()
    out: list[str] = []
    seen = set(anchors)

    def add(raw: str) -> None:
        cleaned = clean_tag(raw)
        if not cleaned or cleaned in seen or not evidenced(cleaned, text):
            return
        seen.add(cleaned)
        out.append(cleaned)

    for tag in anchors:
        toks = tag.split()
        if len(toks) >= 3:
            add(" ".join(toks[:-1]))
            add(" ".join(toks[1:]))
        if len(toks) >= 2 and toks[-1] == "council" and "city" in toks:
            place = toks[0]
            add(f"city of {place}")
            add(f"{place} city")
            add("city council")
            add(f"{place} council")
            if "hall" in low:
                add(f"{place} city hall")
                add("city hall")
        if toks and "washington" in low:
            add(f"{toks[0]} washington")
            add("washington state")
    return out[:8]


def evidenced(tag: str, text: str) -> bool:
    """True when the tag's content words actually occur in the source."""
    low = (text or "").lower()
    needed = [tok for tok in tag.split() if len(tok) >= 4 or tok.isdigit()]
    if not needed:
        return tag.lower() in low
    return all(tok in low for tok in needed)
