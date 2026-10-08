"""Pick tests that sit near a section string and away from each other.

The validator embeds `title: description` and, for each test,
`description` plus `Assertion: {assertion}`, after stripping newlines.
Selection uses the same text and cosine, then drops pairs at or above
the near-duplicate margin (validator penalty fires at 0.95).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any
from typing import Sequence


def cosine(vec_a: Sequence[float] | None, vec_b: Sequence[float] | None) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        na += a * a
        nb += b * b
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def embed_text(test: dict[str, Any]) -> str:
    """Match SkillCoverageEvaluationTaskBundle.format_results embedding text."""
    description = (test.get("description") or "").strip()
    assertion = (test.get("assertion") or "").strip()
    if assertion:
        text = f"{description}\nAssertion: {assertion}".strip()
    else:
        text = description
    return text.replace("\n", " ")


def section_target_text(title: str, description: str) -> str:
    return f"{title}: {description}".replace("\n", " ").strip()


def mmr_indices(
    section: Sequence[float] | None,
    vectors: Sequence[Sequence[float] | None],
    *,
    limit: int,
    dup: float,
) -> list[int]:
    """Greedy MMR: closest to the section first, skip a near-duplicate of a keeper.

    A vector that is missing is kept only when nothing else survived, so a
    failed embed cannot wipe the section.
    """
    if limit <= 0 or not vectors:
        return []

    ranked = sorted(
        range(len(vectors)),
        key=lambda i: cosine(section, vectors[i]),
        reverse=True,
    )
    kept: list[int] = []
    for i in ranked:
        vec = vectors[i]
        if not vec:
            continue
        if any(cosine(vec, vectors[j]) >= dup for j in kept):
            continue
        kept.append(i)
        if len(kept) >= limit:
            break
    if not kept and ranked:
        kept = [ranked[0]]
    return kept


def select_best_pair(
    section: Sequence[float] | None,
    vectors: Sequence[Sequence[float] | None],
    skill: Sequence[float] | None = None,
    *,
    hard_dup: float = 0.94,
) -> list[int]:
    """Keep the two tests with the best local score under the duplicate margin.

    Local score uses the same weights as the validator: 0.6 section cosine,
    0.4 cosine of the skill against the mean of the two test vectors.
    A pair at or above ``hard_dup`` is used only when every pair is that close.
    """
    valid = [i for i, vec in enumerate(vectors) if vec]
    if len(valid) <= 2:
        return valid

    best: list[int] | None = None
    best_key: tuple[float, float, float] | None = None
    for i, j in combinations(valid, 2):
        left = vectors[i]
        right = vectors[j]
        pair = cosine(left, right)
        section_score = (cosine(section, left) + cosine(section, right)) / 2.0
        if len(left) == len(right):
            neighborhood = [(a + b) / 2.0 for a, b in zip(left, right)]
            skill_score = cosine(skill, neighborhood)
        else:
            skill_score = 0.0
        raw = (0.6 * section_score) + (0.4 * skill_score)
        safe = pair < hard_dup
        key = (1.0 if safe else 0.0, raw if safe else raw * 0.5, -pair)
        if best_key is None or key > best_key:
            best_key = key
            best = [i, j]
    if not best:
        return valid[:2]
    best.sort(key=lambda i: cosine(section, vectors[i]), reverse=True)
    return best
