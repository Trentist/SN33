"""Read the validator section map. Do not invent a new one.

Pulls the terms a test should reuse: nouns, verbs, boundaries, errors,
commands. The section_id, title, and description stay exactly as supplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from custom_miner.skill_pipeline.normalize import section_parts

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_BACKTICK = re.compile(r"`([^`]+)`")
_COMMAND = re.compile(r"\b(?:git|npm|pip|docker|kubectl)\s+[A-Za-z0-9_\-]+")
_FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_\-]*")
_ERROR = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:Error|Exception))\b")

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over",
    "when", "then", "than", "each", "per", "its", "are", "was", "has",
    "have", "not", "but", "any", "all", "via", "using", "used", "before",
    "after", "their", "them", "they", "will", "must", "should", "how",
    "what", "which", "where", "while", "during", "about", "only", "also",
}

_VERBS = {
    "abort", "continue", "resolve", "restore", "return", "raise", "parse",
    "convert", "map", "validate", "record", "sign", "stage", "commit",
    "rebase", "slugify", "normalize", "reject", "fail", "resume", "create",
    "delete", "update", "check", "verify", "handle", "apply", "transform",
    "encode", "decode", "format", "compute", "calculate", "draft", "explain",
    "list", "limit", "consent", "port", "clean", "pause", "add", "push",
}

_BOUNDARIES = {
    "empty", "null", "none", "zero", "max", "min", "boundary", "edge",
    "invalid", "missing", "overflow", "blank", "duplicate", "conflict",
    "paused", "abort", "idempotent", "unicode", "accented", "whitespace",
}


@dataclass(frozen=True)
class SectionBrief:
    section_id: str
    title: str
    description: str
    nouns: tuple[str, ...]
    verbs: tuple[str, ...]
    boundaries: tuple[str, ...]
    errors: tuple[str, ...]
    commands: tuple[str, ...]


def _uniq(items: list[str], limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return tuple(out)


def parse_section(section) -> SectionBrief:
    sid, title, desc = section_parts(section)
    blob = f"{title} {desc}"
    words = _WORD.findall(blob)
    verbs = [w for w in words if w.lower() in _VERBS]
    boundaries = [w for w in words if w.lower() in _BOUNDARIES]
    nouns = [
        w for w in words
        if w.lower() not in _STOP and w.lower() not in _VERBS and w.lower() not in _BOUNDARIES
    ]
    errors = _ERROR.findall(blob)
    if re.search(r"\berror|invalid|fail|abort\b", blob, re.I):
        errors = list(errors) + ["error"]
    commands = _BACKTICK.findall(blob) + _COMMAND.findall(blob) + _FLAG.findall(blob)
    return SectionBrief(
        section_id=sid,
        title=title,
        description=desc,
        nouns=_uniq(nouns, 8),
        verbs=_uniq(verbs, 6),
        boundaries=_uniq(boundaries, 6),
        errors=_uniq(errors, 4),
        commands=_uniq(commands, 4),
    )


def parse_section_map(section_map: list | None) -> list[SectionBrief]:
    return [parse_section(section) for section in list(section_map or [])]


def format_section_brief(brief: SectionBrief) -> str:
    """Prompt text. The section_id line is the validator's, unchanged."""
    lines = [f"- [{brief.section_id}] {brief.title}: {brief.description}"]
    if brief.nouns:
        lines.append("  terms: " + ", ".join(brief.nouns))
    if brief.verbs:
        lines.append("  actions: " + ", ".join(brief.verbs))
    if brief.boundaries:
        lines.append("  boundaries: " + ", ".join(brief.boundaries))
    if brief.errors:
        lines.append("  errors: " + ", ".join(brief.errors))
    if brief.commands:
        lines.append("  commands: " + ", ".join(brief.commands))
    return "\n".join(lines)
