"""Local tag cleaning aligned to validator validate_tag_set survival.

A miner tag is scored only when it is already in ``Utils.get_safe_tag`` form:
lowercase, letters and spaces, length 3–50. Hyphens and other punctuation are
rewritten to spaces so the string still matches the validator intersection.
"""

from __future__ import annotations

import re

from conversationgenome.utils.Utils import Utils

_MAX_LEN = 50
_MAX_TOKENS = 6

# Exact phrases and single tokens that fail the validator's "good English
# keyword" check or sit far from any page-topic centroid.
_FLUFF = {
    "home",
    "menu",
    "contact",
    "privacy",
    "cookie",
    "cookies",
    "login",
    "subscribe",
    "newsletter",
    "share",
    "footer",
    "navigation",
    "homepage",
    "website",
    "webpage",
    "article",
    "articles",
    "blog",
    "news",
    "search",
    "click",
    "about us",
    "contact us",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "sign up",
    "log in",
    "read more",
    "click here",
    "best practices",
    "overview",
    "introduction",
    "information",
    "welcome",
    "services",
    "solutions",
    "company",
    "business",
    "technology",
    "software",
    "update",
    "page",
    "site",
    "info",
    "guide",
    "api",
    "examples",
    "example",
    "documentation",
    "troubleshooting",
    "system design",
    "timeout",
    "reliability",
    "monitoring",
}

_URLISH = re.compile(r"(https?:|www\.|\.com\b|\.org\b|\.net\b)", re.I)
_BULLET = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")


def clean_tag(raw: str) -> str | None:
    """Return the validator-safe form, or None when the tag should be dropped."""
    if not raw or not isinstance(raw, str):
        return None
    text = _BULLET.sub("", raw).strip().strip("\"'`")
    if not text or _URLISH.search(text):
        return None
    safe = Utils.get_safe_tag(text)
    if not safe or len(safe) < 3 or len(safe) > _MAX_LEN:
        return None
    if safe in _FLUFF or safe.isdigit():
        return None
    toks = safe.split()
    if not toks or len(toks) > _MAX_TOKENS:
        return None
    if len(toks) == 1:
        tok = toks[0]
        vowels = len(re.findall(r"[aeiou]", tok))
        if len(tok) >= 5 and vowels == 0:
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
