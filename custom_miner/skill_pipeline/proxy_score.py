"""Validator-equivalent score over every hidden 3-section subset.

Assumes tests that passed the local lint would also pass the judge.
The real judge still runs on the validator. This score is only for
choosing which two tests to keep, and for logging the worst trio.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

from custom_miner.skill_pipeline.select import cosine

SECTION_WEIGHT = 0.6
SKILL_WEIGHT = 0.4
VALIDATOR_DUPLICATE = 0.95
TOO_FEW_TESTS = 5
TOO_FEW_PENALTY = 0.5
DUPLICATE_PENALTY = 0.5


def hidden_subsets(section_ids: Sequence[str]) -> list[tuple[str, ...]]:
    ids = [sid for sid in section_ids if sid]
    if len(ids) <= 3:
        return [tuple(ids)] if ids else []
    return list(combinations(ids, 3))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(width)]


def _has_duplicate(vectors: Sequence[Sequence[float]], threshold: float = VALIDATOR_DUPLICATE) -> bool:
    for i, j in combinations(range(len(vectors)), 2):
        if cosine(vectors[i], vectors[j]) >= threshold:
            return True
    return False


def score_subset(
    subset: Sequence[str],
    section_vectors: dict[str, Sequence[float] | None],
    test_vectors: dict[str, Sequence[Sequence[float]]],
    skill_vector: Sequence[float] | None,
    *,
    apply_penalties: bool = True,
) -> float:
    """One hidden trio, using the public 0.6 / 0.4 weights and penalties."""
    if not subset:
        return 0.0
    section_scores: list[float] = []
    used: list[Sequence[float]] = []
    for sid in subset:
        vecs = [v for v in (test_vectors.get(sid) or []) if v]
        section_vec = section_vectors.get(sid)
        if not vecs or not section_vec:
            section_scores.append(0.0)
            continue
        sims = sorted((cosine(section_vec, vec) for vec in vecs), reverse=True)[:2]
        section_scores.append(sum(sims) / len(sims))
        used.extend(vecs)
    if not section_scores or not any(score > 0.0 for score in section_scores):
        return 0.0
    section_coverage = sum(section_scores) / len(section_scores)
    neighborhood = _mean_vector(used)
    skill_coverage = cosine(skill_vector, neighborhood) if neighborhood else 0.0
    score = (SECTION_WEIGHT * section_coverage) + (SKILL_WEIGHT * skill_coverage)
    if apply_penalties and len(used) < TOO_FEW_TESTS:
        score *= TOO_FEW_PENALTY
    if apply_penalties and _has_duplicate(used):
        score *= DUPLICATE_PENALTY
    return score


def max_pairwise(vectors: Sequence[Sequence[float]]) -> float:
    best = 0.0
    for i, j in combinations(range(len(vectors)), 2):
        best = max(best, cosine(vectors[i], vectors[j]))
    return best


def score_candidate(
    section_ids: Sequence[str],
    section_vectors: dict[str, Sequence[float] | None],
    test_vectors: dict[str, Sequence[Sequence[float]]],
    skill_vector: Sequence[float] | None,
) -> dict[str, float | str]:
    """Mean, worst trio, and duplicate margin for one generated bundle."""
    ids = [sid for sid in section_ids if sid]
    scores = [
        score_subset(subset, section_vectors, test_vectors, skill_vector)
        for subset in hidden_subsets(ids)
    ]
    summary: dict[str, float | str] = dict(summarize_hidden_scores(scores))
    flat: list[Sequence[float]] = []
    weakest_sid = ""
    weakest = 1.0
    for sid in ids:
        vecs = [vec for vec in (test_vectors.get(sid) or []) if vec]
        flat.extend(vecs)
        section_vec = section_vectors.get(sid)
        if not vecs or not section_vec:
            mean = 0.0
        else:
            mean = sum(cosine(section_vec, vec) for vec in vecs) / len(vecs)
        if mean < weakest:
            weakest = mean
            weakest_sid = sid
    summary["max_pairwise"] = max_pairwise(flat)
    summary["weakest_section"] = weakest_sid
    summary["weakest_section_cosine"] = weakest if ids else 0.0
    return summary


def summarize_hidden_scores(scores: Sequence[float]) -> dict[str, float]:
    if not scores:
        return {
            "mean": 0.0,
            "minimum": 0.0,
            "p10": 0.0,
            "maximum": 0.0,
            "selection": 0.0,
            "n_subsets": 0.0,
        }
    ordered = sorted(scores)
    mean = sum(ordered) / len(ordered)
    minimum = ordered[0]
    p10 = ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)]
    return {
        "mean": mean,
        "minimum": minimum,
        "p10": p10,
        "maximum": ordered[-1],
        "selection": (0.70 * mean) + (0.30 * minimum),
        "n_subsets": float(len(ordered)),
    }
