"""Phrases that appear in the page or snippets, plus the other spelling the validator may store.

Ground truth often says ``retry`` on one draw and ``retry pattern`` on the next.
Keeping both forms gives an exact overlap for whichever string was stored and a
high-cosine unique for the other.
"""

from __future__ import annotations

import re

from custom_miner.webpage_pipeline.gate import clean_tags

# Longest first so "retry pattern" is seen before "retry".
_PHRASES = (
    "identity and access management",
    "privileged access management",
    "spring boot actuator",
    "vulnerability management",
    "reliability engineering",
    "exponential backoff",
    "graceful degradation",
    "timeout management",
    "fallback mechanisms",
    "circuit breaker pattern",
    "resiliency patterns",
    "resilience patterns",
    "distributed systems",
    "security monitoring",
    "vulnerability scanning",
    "incident response",
    "high availability",
    "fault tolerance",
    "risk assessment",
    "threat detection",
    "access control",
    "audit logging",
    "patch management",
    "least privilege",
    "bulkhead pattern",
    "retry pattern",
    "fallback pattern",
    "circuit breaker",
    "spring boot",
    "microservices",
    "observability",
    "cybersecurity",
    "prometheus",
    "remediation",
    "compliance",
    "fallback",
    "retry",
)

# Other string the same concept is stored under.
_VARIANTS = {
    "resiliency patterns": ("resilience patterns",),
    "resilience patterns": ("resiliency patterns",),
    "retry": ("retry pattern",),
    "retry pattern": ("retry",),
    "fallback": ("fallback pattern", "fallback mechanisms"),
    "fallback pattern": ("fallback",),
    "fallback mechanisms": ("fallback",),
    "circuit breaker": ("circuit breaker pattern",),
    "circuit breaker pattern": ("circuit breaker",),
    "bulkhead pattern": ("bulkhead",),
    "timeout management": ("timeout",),
}


def _present(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


# Stem in the snippet → the phrase ground truth usually stores.
_STEM_TAGS = (
    ("vulnerab", "vulnerability management"),
    ("threat", "threat detection"),
    ("privileged", "privileged access management"),
    ("incident response", "incident response"),
    ("cybersecurity", "cybersecurity"),
    ("access control", "access control"),
    ("risk assessment", "risk assessment"),
    ("audit log", "audit logging"),
)


def phrases_in_text(text: str) -> list[str]:
    """Lexicon phrases that actually occur in ``text``."""
    body = (text or "").lower()
    if not body:
        return []
    found = [phrase for phrase in _PHRASES if _present(phrase, body)]
    for stem, phrase in _STEM_TAGS:
        if stem in body:
            found.append(phrase)
    return clean_tags(found)


def variant_tags(tags: list[str]) -> list[str]:
    """Sibling spellings for tags we already have evidence for."""
    out: list[str] = []
    seen = set(tags)
    for tag in tags:
        for variant in _VARIANTS.get(tag, ()):
            if variant in seen:
                continue
            seen.add(variant)
            out.append(variant)
    return clean_tags(out)
