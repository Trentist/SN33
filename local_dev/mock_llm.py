"""Keyword / heuristic miner for offline local_dev (no API key required)."""

from __future__ import annotations

import re
from typing import Any

from conversationgenome.task.Task import Task

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "am",
    "you", "i", "it", "my", "me", "we", "they", "that", "this", "with", "as", "at",
    "be", "have", "has", "was", "were", "do", "did", "how", "what", "when", "your",
    "too", "one", "only", "get", "got", "oh", "hey", "ugh", "yay", "no", "yes",
}


def _window_text(task: Task) -> str:
    data = getattr(task.input, "data", None)
    if data is None:
        return ""

    # conversation / webpage style windows
    window = getattr(data, "window", None)
    if window:
        return " ".join(line[1] if isinstance(line, (list, tuple)) else str(line) for line in window)

    lines = getattr(data, "lines", None)
    if lines:
        return " ".join(line[1] if isinstance(line, (list, tuple)) else str(line) for line in lines)

    comment = getattr(data, "comment", None) or getattr(data, "free_form_comment", None)
    question = getattr(data, "survey_question", None)
    parts = [p for p in (question, comment) if p]
    if parts:
        return " ".join(parts)

    return str(data)


def _phrase_tags(text: str) -> list[str]:
    """Pull useful multi-word and single-word topical tags from free text."""
    lower = text.lower()
    phrases = []
    # curated phrase hits for the sample dialogue + general usefulness
    phrase_bank = [
        "wedding planning",
        "side hustle",
        "luxury cars",
        "baby boys",
        "middle kid",
        "mobile app",
        "user friendly",
    ]
    for p in phrase_bank:
        if p in lower:
            phrases.append(p)

    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", lower)
    unigrams = []
    for t in tokens:
        if t in _STOP:
            continue
        unigrams.append(t)

    # light topical expansions
    expansions = {
        "wedding": ["wedding planning", "marriage"],
        "babysitting": ["babysitting", "childcare"],
        "kids": ["children", "parenthood"],
        "parent": ["parenthood", "family"],
        "mercedes": ["mercedes", "luxury cars"],
        "ferrari": ["ferrari", "luxury cars"],
        "titanic": ["titanic", "movies"],
        "movie": ["movies"],
    }
    for token in list(unigrams):
        for extra in expansions.get(token, []):
            phrases.append(extra)

    seen = set()
    out = []
    for tag in phrases + unigrams:
        if tag in seen or tag in _STOP:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= 12:
            break
    return out


def _mock_skill_coverage(task: Task) -> dict[str, Any]:
    """Deterministic skill bundle covering every section — for offline sims."""
    data = getattr(task.input, "data", None)
    seed = getattr(data, "seed", None) or "Skill for local_dev mock coverage."
    section_map = list(getattr(data, "section_map", None) or [])

    headings = []
    section_tests: dict[str, list] = {}
    for section in section_map:
        sid = getattr(section, "section_id", "section")
        title = getattr(section, "title", sid)
        desc = getattr(section, "description", title)
        fn = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower() or sid
        err = f"{fn.title().replace('_', '')}Error"
        headings.append(
            f"## {title}\n{desc}\n"
            f"Observable behavior:\n"
            f"- `{fn}(\"Sample Input\")` → `{fn}-sample`\n"
            f"- `{fn}(None)` raises `{err}`\n"
        )
        section_tests[sid] = [
            {
                "name": f"test_{sid}_core",
                "description": f"Verify {title}. {desc} Canonical input.",
                "assertion": f'{fn}("Sample Input") == "{fn}-sample"',
            },
            {
                "name": f"test_{sid}_invalid",
                "description": f"Verify {title} rejects a missing input with {err}.",
                "assertion": f"{fn}(None) raises {err}",
            },
        ]

    if not section_tests:
        section_tests = {
            "s1": [
                {
                    "name": "test_s1_core",
                    "description": "Basic ASCII slugify lowercases and hyphenates.",
                    "assertion": 'slugify("Hello World") == "hello-world"',
                },
                {
                    "name": "test_s1_invalid",
                    "description": "Empty input returns empty slug.",
                    "assertion": 'slugify("") == ""',
                },
            ]
        }
        headings = [
            "## Basic ASCII transformation\n"
            "Implement `slugify` to lowercase and replace whitespace with hyphens.\n"
            "Empty input returns empty string.\n"
        ]

    skill = f"# Mock Skill\n\n{seed}\n\n" + "\n".join(headings)
    return {
        "skill": skill,
        "tdd_plan": "Two concrete checks per section, both written into the skill.",
        "section_tests": section_tests,
        "seed": seed if isinstance(seed, str) else "",
    }


async def mock_mine(task: Task) -> dict[str, Any]:
    if task.type == "skill_coverage_evaluation":
        from custom_miner.skill_pipeline import normalize_skill_bundle

        raw = _mock_skill_coverage(task)
        section_map = getattr(getattr(task.input, "data", None), "section_map", None)
        return normalize_skill_bundle(raw, section_map)

    tags = _phrase_tags(_window_text(task))
    if len(tags) < 3:
        tags = (tags + ["general discussion", "daily life", "personal update"])[:5]
    return {"tags": tags, "vectors": None}
