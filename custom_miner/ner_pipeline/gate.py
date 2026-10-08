"""Clean named-entity tags the way the validator will score them.

``validate_named_entities_tag_set`` only runs ``Utils.get_clean_tag_set``:
lowercase, letters and digits, length 3–64. Twenty or more tags are randomly
cut down to 20, so this gate also rejects filler that would survive that cut
and then sit far from the entity centroid.
"""

from __future__ import annotations

import re

from conversationgenome.utils.Utils import Utils

_MAX_LEN = 64
_MAX_TOKENS = 8

_FILLER = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "was",
    "were",
    "said",
    "says",
    "also",
    "into",
    "about",
    "their",
    "they",
    "them",
    "his",
    "her",
    "she",
    "you",
    "your",
    "our",
    "not",
    "but",
    "are",
    "been",
    "being",
    "its",
    "it's",
    "transcript",
    "speaker",
    "unknown",
    "none",
    "null",
    "n a",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "mayor",
    "president",
    "senator",
    "governor",
    "people",
    "person",
    "organization",
    "organisations",
    "organizations",
    "location",
    "locations",
    "entity",
    "entities",
    "named entity",
    "named entities",
    "information",
    "overview",
    "introduction",
    "article",
    "news",
    "update",
}

# Sentence glue that leaks in when a span crosses a line break.
# Titles such as "mayor" stay, because "mayor adams" is often the ground-truth string.
_FUNCTION = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "was",
    "were",
    "said",
    "says",
    "also",
    "into",
    "about",
    "their",
    "they",
    "them",
    "his",
    "her",
    "she",
    "you",
    "your",
    "our",
    "not",
    "but",
    "are",
    "been",
    "being",
    "its",
}

# Roles and header words that stick to a real name and knock it off the ground-truth string.
_EDGE = _FUNCTION | {
    "member",
    "members",
    "chair",
    "chairperson",
    "vice",
    "present",
    "agenda",
    "report",
    "proceedings",
    "journal",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "monday",
    "saturday",
    "sunday",
    "excused",
}

_SPEAKER = re.compile(r"^speaker\s*\d*$")
_BULLET = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")


def clean_tag(raw: str) -> str | None:
    """Return the validator-safe form, or None when the tag should be dropped."""
    if not raw or not isinstance(raw, str):
        return None
    text = _BULLET.sub("", raw).strip().strip("\"'`[]")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    safe = Utils.get_safe_tag(text)
    if not safe or _SPEAKER.match(safe):
        return None
    toks = [tok for tok in safe.split() if tok]
    while toks and toks[0] in _EDGE:
        toks = toks[1:]
    while toks and toks[-1] in _EDGE:
        toks = toks[:-1]
    if not toks or len(toks) > _MAX_TOKENS:
        return None
    if all(tok in _FILLER for tok in toks):
        return None
    safe = " ".join(toks)
    if len(safe) < 3 or len(safe) > _MAX_LEN or safe.isdigit() or safe in _FILLER:
        return None
    if any(tok in {"avenue", "ave", "street", "st"} for tok in toks):
        return None
    return safe


def clean_tags(tags: list[str]) -> list[str]:
    """Dedupe in order, keeping the first occurrence of each safe tag."""
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = clean_tag(tag)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out
