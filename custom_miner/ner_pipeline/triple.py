"""Three-tag named-entity selector.

The validator score is dominated by the top three tags that are not exact
ground-truth strings. This module builds a high-recall anchor pool, simulates
the validator's combine step, and picks the triple of non-verbatim phrases
that maximizes

    0.7 * expected score + 0.3 * 25th-percentile score

across those simulations. An exact match is not rejected up front. It loses
the 55% slot only in the simulations that actually contain that string.
"""

from __future__ import annotations

import itertools
import random
import re
import zlib
from dataclasses import dataclass

import numpy as np

from custom_miner.ner_pipeline.aliases import extract_spans
from custom_miner.ner_pipeline.gate import clean_tag

_INSTITUTION = {
    "council",
    "hall",
    "committee",
    "commission",
    "bill",
    "ordinance",
    "department",
    "levy",
    "charter",
    "plan",
    "law",
    "act",
    "grant",
    "project",
}
_JUNK = {
    "report",
    "agenda",
    "proceedings",
    "journal",
    "please",
    "note",
    "officer",
    "rules",
    "procedures",
    "comment",
    "approval",
    "meeting",
    "present",
    "excused",
    "avenue",
    "ave",
    "street",
    "floor",
    "page",
    "item",
    "call",
    "order",
}
_GENERIC = {
    "city",
    "county",
    "state",
    "council",
    "committee",
    "commission",
    "department",
    "hall",
    "bill",
    "law",
    "act",
    "grant",
    "project",
    "plan",
    "local",
    "public",
    "municipal",
    "government",
    "administration",
}


@dataclass(frozen=True)
class TripleChoice:
    tags: list[str]
    expected: float
    lower: float
    objective: float


def validator_score(cosines: list[float], exact: list[bool]) -> float:
    """Validator NER score for one triple against one ground-truth set.

    Exact strings are removed from the top-three unique term. Missing unique
    slots are zeros. Mean, median, and max still use every tag.
    """
    if not cosines:
        return 0.0
    unique = [value for value, is_exact in zip(cosines, exact) if not is_exact]
    top = sorted(unique, reverse=True)[:3]
    while len(top) < 3:
        top.append(0.0)
    mid = float(np.median(cosines))
    return float(
        0.55 * (sum(top) / 3.0)
        + 0.25 * (sum(cosines) / len(cosines))
        + 0.10 * mid
        + 0.10 * max(cosines)
    )


def _strip_junk(tag: str) -> str | None:
    """Drop glue words without discarding the institution they were attached to."""
    toks = [tok for tok in tag.split() if tok not in _JUNK]
    if not toks:
        return None
    if len(toks) == 1 and not any(ch.isdigit() for ch in toks[0]):
        return None
    return " ".join(toks)


def anchors_from_text(text: str) -> list[str]:
    """High-recall proper names, bills, and institutions from one text."""
    out: list[str] = []
    seen: set[str] = set()
    for span in extract_spans(text or ""):
        kept = _strip_junk(span)
        if not kept or kept in seen:
            continue
        seen.add(kept)
        out.append(kept)
    return out[:40]


def _is_person(tag: str) -> bool:
    toks = tag.split()
    return 2 <= len(toks) <= 3 and not any(tok in _INSTITUTION or tok == "city" or tok.isdigit() for tok in toks)


def _is_institution(tag: str) -> bool:
    return any(tok in _INSTITUTION or tok == "city" for tok in tag.split())


def candidate_pool(anchors: list[str], bridges: list[str]) -> list[str]:
    """Bridges first, then institutional anchors. People stay in the centroid only."""
    out: list[str] = []
    seen: set[str] = set()
    for tag in list(bridges) + [tag for tag in anchors if _is_institution(tag)]:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= 18:
            break
    return out


_CITY_STATE = re.compile(r"\b([A-Z][a-z]{3,})\s*,\s*[A-Z]{2}\b")
_CIVIC_HEAD = {"city", "council", "hall"}


def _surnames(anchors: list[str]) -> set[str]:
    names: set[str] = set()
    for tag in anchors:
        if _is_person(tag):
            names.add(tag.split()[-1])
    return names


def _place_hint(text: str) -> str | None:
    """City in ``Seattle, WA``. That pattern is the place, not a repeated surname."""
    counts: dict[str, int] = {}
    for match in _CITY_STATE.finditer(text or ""):
        tok = match.group(1).lower()
        if tok in _GENERIC or tok in _INSTITUTION:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda tok: counts[tok])


def _dominant_place(anchors: list[str]) -> str | None:
    """Distinctive token inside institutional anchors, excluding person surnames."""
    blocked = _surnames(anchors)
    counts: dict[str, int] = {}
    for tag in anchors:
        if not _is_institution(tag) or _is_person(tag):
            continue
        weight = 3 if any(tok in _CIVIC_HEAD for tok in tag.split()) else 1
        for tok in tag.split():
            if len(tok) < 5 or tok in _GENERIC or tok in _INSTITUTION or tok in blocked:
                continue
            counts[tok] = counts.get(tok, 0) + weight
    if not counts:
        return None
    return max(counts, key=lambda tok: (counts[tok], len(tok)))


def bridge_candidates(anchors: list[str], place: str | None = None) -> list[str]:
    """Institutional paraphrases of the anchors. These need not occur verbatim."""
    place = place or _dominant_place(anchors)
    raw: list[str] = []
    if place:
        raw.extend(
            [
                f"{place} municipal government",
                f"{place} city government",
                f"{place} local government",
                f"{place} city administration",
                f"{place} municipal council",
                f"{place} city legislature",
            ]
        )
        kinds = {tok for tag in anchors for tok in tag.split()}
        if "hall" in kinds:
            raw.append(f"{place} municipal offices")
        if "bill" in kinds or "ordinance" in kinds or "law" in kinds:
            raw.append(f"{place} municipal legislation")
        if "levy" in kinds:
            raw.append(f"{place} public levy")
        if "committee" in kinds or "commission" in kinds:
            raw.append(f"{place} civic committee")
        if "housing" in kinds:
            raw.append(f"{place} housing policy")
    anchor_set = set(anchors)
    out: list[str] = []
    seen: set[str] = set()
    for phrase in raw:
        cleaned = clean_tag(phrase)
        if not cleaned or cleaned in seen or cleaned in anchor_set:
            continue
        if len(cleaned.split()) < 2:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _collapse(tags: list[str], rng: random.Random) -> list[str]:
    kept: list[str] = []
    for tag in tags:
        toks = set(tag.split())
        merged = False
        for index, prev in enumerate(kept):
            prev_toks = set(prev.split())
            if not toks or not prev_toks:
                continue
            overlap = len(toks & prev_toks) / len(toks | prev_toks)
            if overlap < 0.5 and not (toks <= prev_toks or prev_toks <= toks):
                continue
            longer = tag if len(tag) >= len(prev) else prev
            shorter = prev if longer == tag else tag
            kept[index] = longer if rng.random() < 0.7 else shorter
            merged = True
            break
        if not merged:
            kept.append(tag)
    return kept


def simulate_ground_truths(
    transcript_anchors: list[str],
    snippet_groups: list[list[str]],
    *,
    n: int = 48,
    seed: int = 0,
) -> list[set[str]]:
    """Omit some entities, collapse aliases, and cap at 20.

    Snippets the miner can see were already chosen by the validator, so every
    simulation may use them. People are omitted more often than institutions,
    which is the combine step's main source of movement.
    """
    rng = random.Random(seed)
    pool = list(transcript_anchors)
    for group in snippet_groups:
        pool.extend(group)
    if not pool:
        return []
    sets: list[set[str]] = []
    for _ in range(n):
        chosen: list[str] = []
        for tag in pool:
            drop_p = 0.55 if _is_person(tag) else 0.25
            if rng.random() < drop_p:
                continue
            chosen.append(tag)
        if len(chosen) < 4:
            chosen = list(pool[:8])
        collapsed = _collapse(chosen, rng)
        if len(collapsed) > 20:
            people = [tag for tag in collapsed if _is_person(tag)]
            others = [tag for tag in collapsed if tag not in people]
            rng.shuffle(people)
            collapsed = (others + people)[:20]
        sets.append(set(collapsed))
    return sets


def _cosine_matrix(candidates: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    cand = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
    cent = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    return cand @ cent.T


def _objective(scores: list[float]) -> tuple[float, float, float]:
    expected = float(np.mean(scores))
    lower = float(np.percentile(scores, 25))
    return expected, lower, 0.7 * expected + 0.3 * lower


def simulation_centroids(
    simulations: list[set[str]],
    vectors: dict[str, np.ndarray],
    *,
    centroid_shift: np.ndarray | None = None,
) -> list[tuple[set[str], np.ndarray]]:
    """One centroid per simulated ground-truth set that has at least two vectors."""
    out: list[tuple[set[str], np.ndarray]] = []
    for sim in simulations:
        rows = [vectors[tag] for tag in sim if tag in vectors]
        if len(rows) < 2:
            continue
        centroid = np.mean(np.stack(rows, axis=0), axis=0)
        if centroid_shift is not None:
            centroid = centroid + centroid_shift
        out.append((sim, centroid.astype(np.float32)))
    return out


def select_triple(
    candidates: list[str],
    vectors: dict[str, np.ndarray],
    simulations: list[set[str]],
    *,
    centroid_shift: np.ndarray | None = None,
) -> TripleChoice | None:
    """Pick the triple with the best exact-aware score across simulations.

    A tag that is exact in one simulation loses the 55% slot in that simulation
    only. It is not removed from the search.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in candidates:
        if tag in seen or tag not in vectors:
            continue
        seen.add(tag)
        ordered.append(tag)
    ordered = ordered[:18]
    packed = simulation_centroids(simulations, vectors, centroid_shift=centroid_shift)
    if len(ordered) < 3 or len(packed) < 4:
        return None
    matrix = np.stack([vectors[tag] for tag in ordered], axis=0).astype(np.float32)
    centroids = np.stack([centroid for _, centroid in packed], axis=0)
    sims = _cosine_matrix(matrix, centroids)
    exact = np.zeros(sims.shape, dtype=bool)
    for column, (members, _) in enumerate(packed):
        for row, tag in enumerate(ordered):
            if tag in members:
                exact[row, column] = True
    combos = np.asarray(list(itertools.combinations(range(len(ordered)), 3)), dtype=np.int32)
    chosen = sims[combos]  # (triples, 3, simulations)
    chosen_exact = exact[combos]
    # Exact tags drop out of the top-3 unique term. Missing slots are zeros.
    ranked = np.sort(np.where(chosen_exact, -np.inf, chosen), axis=1)[:, ::-1, :]
    top3_mean = np.where(np.isfinite(ranked), ranked, 0.0).mean(axis=1)
    mean = chosen.mean(axis=1)
    median = np.median(chosen, axis=1)
    maximum = chosen.max(axis=1)
    scores = 0.55 * top3_mean + 0.25 * mean + 0.10 * median + 0.10 * maximum
    expected = scores.mean(axis=1)
    lower = np.percentile(scores, 25, axis=1)
    objective = 0.7 * expected + 0.3 * lower
    winner = int(np.argmax(objective))
    return TripleChoice(
        tags=[ordered[int(index)] for index in combos[winner]],
        expected=float(expected[winner]),
        lower=float(lower[winner]),
        objective=float(objective[winner]),
    )


def best_actual_triple(
    candidates: list[str],
    vectors: dict[str, np.ndarray],
    true_centroid: np.ndarray,
    true_tags: set[str],
) -> tuple[list[str], float] | None:
    """Highest real score inside the same candidate pool the selector searches."""
    ordered: list[str] = []
    seen: set[str] = set()
    for tag in candidates:
        if tag in seen or tag not in vectors:
            continue
        seen.add(tag)
        ordered.append(tag)
        if len(ordered) >= 18:
            break
    best_tags: list[str] | None = None
    best_score = -1.0
    for combo in itertools.combinations(ordered, 3):
        score = score_against(list(combo), vectors, true_centroid, true_tags)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_tags = list(combo)
    if best_tags is None:
        return None
    return best_tags, best_score


def score_against(
    tags: list[str],
    vectors: dict[str, np.ndarray],
    true_centroid: np.ndarray,
    true_tags: set[str],
) -> float | None:
    """Score a triple against one stored ground truth."""
    cosines: list[float] = []
    exact: list[bool] = []
    denom = float(np.linalg.norm(true_centroid))
    if denom < 1e-12:
        return None
    for tag in tags:
        vec = vectors.get(tag)
        if vec is None:
            return None
        cosines.append(float(np.dot(true_centroid, vec) / (denom * np.linalg.norm(vec))))
        exact.append(tag in true_tags)
    return validator_score(cosines, exact)


def prepare_document(transcript: str, snippets: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """Return transcript anchors, per-snippet anchors, and bridge candidates."""
    transcript_anchors = anchors_from_text(transcript)
    groups = [anchors_from_text(snippet) for snippet in snippets]
    pool: list[str] = []
    seen: set[str] = set()
    for tag in transcript_anchors + [tag for group in groups for tag in group]:
        if tag not in seen:
            seen.add(tag)
            pool.append(tag)
    place = _place_hint(transcript) or _dominant_place(pool)
    bridges = bridge_candidates(pool, place)
    return transcript_anchors, groups, bridges


def mean_centroid(tags: list[str], vectors: dict[str, np.ndarray]) -> np.ndarray | None:
    rows = [vectors[tag] for tag in tags if tag in vectors]
    if len(rows) < 2:
        return None
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def document_seed(transcript: str, snippets: list[str]) -> int:
    blob = transcript + "\n" + "\n".join(snippets)
    return zlib.adler32(blob.encode("utf-8", errors="ignore")) & 0xFFFFFFFF


def load_stored_ground_truth(bundle: dict) -> tuple[np.ndarray | None, set[str]]:
    """Centroid and safe tag set from a prepared bundle."""
    meta = (bundle.get("input") or {}).get("metadata") or {}
    raw_vectors = meta.get("vectors") or {}
    rows: list[np.ndarray] = []
    tags: set[str] = set()
    for _raw_tag, payload in raw_vectors.items():
        vec = payload.get("vectors") if isinstance(payload, dict) else payload
        if not vec:
            continue
        rows.append(np.asarray(vec, dtype=np.float32))
    # Exact match uses the stored tag strings. The scorer does not re-clean them.
    for raw_tag in meta.get("tags") or []:
        text = str(raw_tag).strip()
        if text:
            tags.add(text)
    if len(rows) < 2:
        return None, tags
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32), tags
