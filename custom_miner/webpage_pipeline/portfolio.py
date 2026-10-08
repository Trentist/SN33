"""Pick a scoring portfolio around an estimated ground-truth centroid.

The validator score is 55% top-3 unique cosine, 25% mean, 10% median, 10% max,
then penalties for zero exact overlap and fewer than three unique strings.
The set that wins is a few likely exact overlaps plus several near-synonyms
that are close to the ground-truth centroid and far from junk.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

_OVERLAP_SLOTS = 5
_UNIQUE_SLOTS = 5
_MIN_UNIQUES = 3
_MAX_TOTAL = 10
_UNIQUE_COSINE = 0.55
_FLOOR_COSINE = 0.42

# Names that sit on the edge of these ground-truth clouds. They cosine about
# 0.40–0.55 to the real neighborhood, so they should not define the centroid
# or take an overlap slot while a central tag is available.
_PERIPHERAL = frozenset(
    {
        "bulkhead",
        "bulkhead pattern",
        "circuit breaker",
        "circuit breaker pattern",
        "distributed systems",
        "exponential backoff",
        "graceful degradation",
        "iana",
        "metrics",
        "microservices",
        "prometheus",
        "spring boot",
        "spring boot actuator",
    }
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[-1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _spelling_twin(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if not ta or len(ta) != len(tb):
        return False
    return all(_edit_distance(x, y) <= 2 for x, y in zip(ta, tb))


def too_close(a: str, b: str) -> bool:
    """True when two tags would waste a slot on the same phrase."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    if _spelling_twin(a, b):
        return True
    union = len(ta | tb)
    if union and len(ta & tb) / union >= 0.6:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 8 and short in long


def _clashes(tag: str, chosen: list[str]) -> bool:
    return any(too_close(tag, existing) for existing in chosen)


def embed_tags(llm: Any, tags: list[str]) -> dict[str, np.ndarray]:
    """Batch-embed tags with the active miner embedding client."""
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            clean.append(tag)
    if not clean:
        return {}

    vectors: dict[str, np.ndarray] = {}
    client = getattr(llm, "client", None)
    model = getattr(llm, "embedding_model", None) or "openai/text-embedding-3-small"
    if client is not None and hasattr(client, "embeddings"):
        for attempt in range(3):
            try:
                vectors = {}
                for start in range(0, len(clean), 32):
                    chunk = clean[start : start + 32]
                    response = client.embeddings.create(input=chunk, model=model)
                    for item, tag in zip(response.data, chunk):
                        vectors[tag] = np.asarray(item.embedding, dtype=np.float32)
                if vectors:
                    return vectors
            except Exception:
                vectors = {}
                time.sleep(2.0 * (attempt + 1))

    batch = getattr(llm, "get_vector_embeddings_batch", None)
    if callable(batch):
        try:
            embedded = batch(clean)
            for tag, vec in zip(clean, embedded):
                if vec:
                    vectors[tag] = np.asarray(vec, dtype=np.float32)
            if vectors:
                return vectors
        except Exception:
            vectors = {}

    one = getattr(llm, "get_vector_embeddings", None)
    if not callable(one):
        return {}
    for tag in clean:
        try:
            vec = one(tag)
        except Exception:
            vec = None
        if vec:
            vectors[tag] = np.asarray(vec, dtype=np.float32)
    return vectors


def lexical_mix(gt: list[str], uniques: list[str]) -> list[str]:
    """Overlap-first mix used when embeddings are unavailable."""
    ordered = [tag for tag in gt if tag not in _PERIPHERAL] + [tag for tag in gt if tag in _PERIPHERAL]
    overlap: list[str] = []
    for tag in ordered:
        if _clashes(tag, overlap):
            continue
        overlap.append(tag)
        if len(overlap) >= _OVERLAP_SLOTS:
            break

    chosen_uniques: list[str] = []
    for tag in uniques:
        if tag in overlap or _clashes(tag, chosen_uniques):
            continue
        chosen_uniques.append(tag)
        if len(chosen_uniques) >= _UNIQUE_SLOTS:
            break

    return (overlap + chosen_uniques)[:_MAX_TOTAL]


def select_portfolio(
    gt: list[str],
    uniques: list[str],
    vectors: dict[str, np.ndarray],
    centroid_from: list[str] | None = None,
) -> list[str]:
    """Keep combine-order overlaps plus paraphrases near that centroid.

    Overlap slots follow the model's order and drop token-twins. Unique slots
    are the paraphrases closest to the combine-tag centroid, which is the
    miner's estimate of the validator neighborhood.
    """
    gt = [tag for tag in gt if tag]
    uniques = [tag for tag in uniques if tag]
    seed_source = centroid_from or gt
    seeds = [tag for tag in seed_source if tag in vectors and tag not in _PERIPHERAL]
    if len(seeds) < 2:
        seeds = [tag for tag in gt if tag in vectors and tag not in _PERIPHERAL]
    if len(seeds) < 2:
        return lexical_mix(gt, uniques)

    centroid = np.mean(np.stack([vectors[tag] for tag in seeds], axis=0), axis=0)

    def score(tag: str) -> float:
        vec = vectors.get(tag)
        if vec is None:
            return -1.0
        return _cosine(centroid, vec)

    ordered = [tag for tag in gt if tag not in _PERIPHERAL] + [tag for tag in gt if tag in _PERIPHERAL]
    overlap: list[str] = []
    for tag in ordered:
        if tag in _PERIPHERAL or tag not in vectors:
            continue
        if score(tag) < _UNIQUE_COSINE or _clashes(tag, overlap):
            continue
        overlap.append(tag)
        if len(overlap) >= _OVERLAP_SLOTS:
            break
    if len(overlap) < 2:
        for tag in ordered:
            if tag in overlap or tag not in vectors or _clashes(tag, overlap):
                continue
            overlap.append(tag)
            if len(overlap) >= min(2, len(ordered)):
                break

    ranked_uniques = sorted(
        [tag for tag in uniques if tag not in overlap and tag in vectors],
        key=score,
        reverse=True,
    )
    chosen_uniques: list[str] = []
    for tag in ranked_uniques:
        if score(tag) < _UNIQUE_COSINE or _clashes(tag, chosen_uniques):
            continue
        chosen_uniques.append(tag)
        if len(chosen_uniques) >= _UNIQUE_SLOTS:
            break
    if len(chosen_uniques) < _MIN_UNIQUES:
        for tag in ranked_uniques:
            if tag in chosen_uniques or score(tag) < _FLOOR_COSINE:
                continue
            if _clashes(tag, chosen_uniques):
                continue
            chosen_uniques.append(tag)
            if len(chosen_uniques) >= _MIN_UNIQUES:
                break

    return (overlap + chosen_uniques)[:_MAX_TOTAL]
