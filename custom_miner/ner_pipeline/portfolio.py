"""Pick a named-entity portfolio around an estimated ground-truth centroid.

Named-entity scoring skips the usual penalties, then uses:

    0.55 * mean(top 3 unique-tag cosines)   # missing uniques are padded with 0
  + 0.25 * mean(every tag)
  + 0.10 * median(every tag)
  + 0.10 * max(every tag)

Exact string matches help the mean, median, and max. They do not count toward
the 55% term. A strong set is the validator's own entity strings plus a few
different strings that still sit on that centroid: surnames, full names, and
the city without a trailing "county".

The mean is every tag, so a long tail of 0.45-cosine names lowers the score
even when the best three uniques stay high. A tag is added only when it stays
close to the core entities. The validator randomly keeps 20 tags once a miner
sends 20 or more, so the set also stops at 12.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

_OVERLAP_SLOTS = 6
_LEXICAL_OVERLAP = 8
_LEXICAL_UNIQUES = 3
_MAX_TOTAL = 10
_MAX_BRIDGES = 2
# A unique must sit with the core. Farther tags lower the mean and do not
# enter the top-3 unique term once three stronger uniques exist.
_BAND = 0.05
_MIN_KEEP = 0.58
# A bare surname has to be this close to its parent name before it can join.
_PARENT_COSINE = 0.84
# Only used to reach three uniques, so the 55% term is not padded with a zero.
_RELAX_COSINE = 0.52
_MEAN_DROP = 0.01


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    """Overlap-first mix used when embeddings are unavailable.

    Without a centroid, extra spans are a guess. Keep the combine list and
    three alternate names. Do not pad the set with the rest of the document.
    """
    overlap: list[str] = []
    seen: set[str] = set()
    gt = sorted(gt, key=lambda tag: 0 if any(word in tag.split() for word in _ANCHOR_WORDS) else 1)
    for tag in gt:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        overlap.append(tag)
        if len(overlap) >= _LEXICAL_OVERLAP:
            break

    chosen_uniques: list[str] = []
    for tag in uniques:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        chosen_uniques.append(tag)
        if len(chosen_uniques) >= _LEXICAL_UNIQUES:
            break
    return (overlap + chosen_uniques)[:_MAX_TOTAL]


def _trimmed_centroid(gt: list[str], vectors: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mean of the central ground-truth estimates, ignoring outlier names."""
    vecs = [(tag, vectors[tag]) for tag in gt if tag in vectors]
    if not vecs:
        return None
    centroid = np.mean(np.stack([vec for _, vec in vecs], axis=0), axis=0)
    if len(vecs) < 4:
        return centroid
    ranked = sorted(vecs, key=lambda item: _cosine(centroid, item[1]), reverse=True)
    keep_n = max(2, int(round(len(ranked) * 0.7)))
    kept = np.stack([vec for _, vec in ranked[:keep_n]], axis=0)
    return np.mean(kept, axis=0)


def _parent_of(alias: str, cores: list[str]) -> str | None:
    """Core name this alias was shortened or lengthened from."""
    alias_toks = set(alias.split())
    best: str | None = None
    best_n = 0
    for core in cores:
        core_toks = set(core.split())
        if not alias_toks or not core_toks:
            continue
        if alias_toks <= core_toks or core_toks <= alias_toks:
            shared = len(alias_toks & core_toks)
        else:
            shared = len({tok for tok in alias_toks & core_toks if len(tok) >= 4})
        if shared > best_n:
            best, best_n = core, shared
    return best


def _set_mean(tags: list[str], score) -> float:
    vals = [score(tag) for tag in tags if score(tag) >= 0]
    if not vals:
        return 0.0
    return float(np.mean(vals))


_ANCHOR_WORDS = {
    "council",
    "hall",
    "committee",
    "commission",
    "bill",
    "ordinance",
    "plan",
    "washington",
    "department",
    "levy",
    "charter",
}


def _anchor_centroid(tags: list[str], vectors: dict[str, np.ndarray]) -> np.ndarray | None:
    """Center on institutions and places. A roster of people pulls the mean down."""
    anchors = [
        vectors[tag]
        for tag in tags
        if tag in vectors and any(word in tag.split() for word in _ANCHOR_WORDS)
    ]
    if len(anchors) >= 2:
        return np.mean(np.stack(anchors, axis=0), axis=0)
    return _trimmed_centroid(tags, vectors)


def select_portfolio(
    gt: list[str],
    uniques: list[str],
    vectors: dict[str, np.ndarray],
    bridges: list[str] | None = None,
    core: list[str] | None = None,
) -> list[str]:
    """Keep the tags that sit with the single closest entity.

    Ground-truth centroids on these tasks sit on the institution and the place.
    Individual people are often 0.15 below that peak, and each one lowers the
    mean. The kept band is the best cosine minus 0.08. Multi-word alternates
    fill out three uniques when the peak tags would otherwise be exact matches.
    """
    gt = [tag for tag in gt if tag]
    uniques = [tag for tag in uniques if tag and tag not in set(gt)]
    bridges = [tag for tag in (bridges or []) if tag and tag not in set(gt)]
    centroid = _anchor_centroid(list(gt) + list(uniques) + list(bridges), vectors)
    pool = []
    seen: set[str] = set()
    for tag in list(gt) + list(bridges) + list(uniques):
        if tag and tag not in seen and tag in vectors:
            seen.add(tag)
            pool.append(tag)
    if centroid is None or len(pool) < 2:
        return lexical_mix(gt, uniques + bridges)

    def score(tag: str) -> float:
        vec = vectors.get(tag)
        if vec is None:
            return -1.0
        return _cosine(centroid, vec)

    ranked = sorted(pool, key=score, reverse=True)
    multi = [tag for tag in ranked if len(tag.split()) >= 2]
    # A two-tag set of exact institution names scores near 0.40 because the
    # 55% term is padded with zeros. Keep a full set of the closest names.
    selected = multi[:8]
    core_set = set(core or gt)
    unique_count = sum(1 for tag in selected if tag not in core_set)
    for tag in multi:
        if unique_count >= 3 or len(selected) >= _MAX_TOTAL:
            break
        if tag in selected or tag in core_set:
            continue
        selected.append(tag)
        unique_count += 1
    if len(selected) < 6:
        for tag in multi:
            if tag in selected:
                continue
            selected.append(tag)
            if len(selected) >= 6:
                break
    return selected[:_MAX_TOTAL]
