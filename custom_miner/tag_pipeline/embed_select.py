"""Embedding-aware tag selection aligned to SN33 GroundTruthTagSimilarity scoring.

Recipe that reaches live 0.85-band when GT neighborhood is coherent:
  - 3–4 evidenced anchor tags (exact-overlap fuel; avoids no_both penalty)
  - 6–8 UNIQUE near-miss tags with HIGH cosine to the anchor neighborhood
    (drives the 55% top_3_unique_mean term)

Uses the same embedding model family validators use (text-embedding-3-small via
the active LlmLib backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable

import numpy as np

from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import normalize_tag


@dataclass
class ScoredTag:
    tag: str
    source: str
    cosine: float
    evidenced: bool


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embed_tags(llm: Any, tags: list[str]) -> dict[str, np.ndarray]:
    """Batch-embed tags; return map normalized_tag -> vector."""
    clean = filter_validate_safe(tags)
    if not clean:
        return {}

    vectors: dict[str, np.ndarray] = {}

    # Prefer native batch when client exists (OpenRouter/OpenAI)
    client = getattr(llm, "client", None)
    model = getattr(llm, "embedding_model", None) or "openai/text-embedding-3-small"
    if client is not None and hasattr(client, "embeddings"):
        try:
            # chunk to keep request size reasonable
            for i in range(0, len(clean), 32):
                chunk = clean[i : i + 32]
                resp = client.embeddings.create(input=chunk, model=model)
                for item, tag in zip(resp.data, chunk):
                    vectors[tag] = np.asarray(item.embedding, dtype=np.float32)
            return vectors
        except Exception:
            pass

    # Fallback: one-by-one via LlmLib
    for tag in clean:
        try:
            vec = llm.get_vector_embeddings(tag)
        except Exception:
            vec = None
        if vec:
            vectors[tag] = np.asarray(vec, dtype=np.float32)
    return vectors


def _is_evidenced(tag: str, text_l: str) -> bool:
    if not text_l:
        return False
    if tag in text_l:
        return True
    toks = tag.split()
    if len(toks) >= 2 and all(t in text_l for t in toks[:2]):
        return True
    # partial: majority of tokens present
    hits = sum(1 for t in toks if len(t) > 2 and t in text_l)
    return hits >= max(2, len(toks) - 1)


def select_embed_aware(
    *,
    llm: Any,
    candidates: list[tuple[str, str]],  # (tag, source)
    combined_text: str,
    n_anchors: int = 4,
    n_uniques: int = 8,
    min_unique_cosine: float = 0.42,
    min_total: int = 8,
    max_total: int = 12,
) -> list[str]:
    """Pick validate-safe tags using cosine to evidenced-anchor neighborhood."""
    text_l = (combined_text or "").lower()

    # Dedupe candidates (preserve first source)
    seen: set[str] = set()
    pool: list[tuple[str, str]] = []
    for tag, source in candidates:
        n = normalize_tag(tag)
        if not n or n in seen:
            continue
        if n not in filter_validate_safe([n]):
            continue
        seen.add(n)
        pool.append((n, source))

    if not pool:
        return []

    vectors = _embed_tags(llm, [t for t, _ in pool])
    if not vectors:
        # Embedding failed — fall back to evidenced+source priority
        evidenced = [t for t, s in pool if _is_evidenced(t, text_l)]
        rest = [t for t, _ in pool if t not in evidenced]
        return (evidenced + rest)[:max_total]

    scored: list[ScoredTag] = []
    for tag, source in pool:
        vec = vectors.get(tag)
        if vec is None:
            continue
        scored.append(
            ScoredTag(
                tag=tag,
                source=source,
                cosine=0.0,  # filled after neighborhood
                evidenced=_is_evidenced(tag, text_l),
            )
        )

    if not scored:
        return []

    # Neighborhood from DIVERSE evidenced seeds (avoid collapsing to one entity cluster)
    evidenced = [s for s in scored if s.evidenced]
    # Prefer person/place-like LLM+bank tags for seeds
    seed_pool = sorted(
        evidenced,
        key=lambda s: (
            0 if s.source in ("llm", "bank") else 1,
            0 if s.source == "anchor" else 1,
            -len(s.tag.split()),
        ),
    )
    neigh_seed: list[ScoredTag] = []
    for s in seed_pool:
        stoks = set(s.tag.split())
        # Diversify: skip if too similar to an already chosen seed
        if any(
            (len(stoks & set(t.tag.split())) / max(len(stoks | set(t.tag.split())), 1)) > 0.5
            for t in neigh_seed
        ):
            continue
        neigh_seed.append(s)
        if len(neigh_seed) >= max(5, n_anchors + 1):
            break
    if len(neigh_seed) < 2:
        neigh_seed = (evidenced or scored)[: min(6, len(evidenced or scored))]

    neigh = np.mean([vectors[s.tag] for s in neigh_seed], axis=0)

    for s in scored:
        s.cosine = _cosine(neigh, vectors[s.tag])

    def _too_close(tag: str, chosen: list[str], thr: float = 0.40) -> bool:
        from custom_miner.tag_pipeline.validate_safe import near_dupe

        return any(near_dupe(tag, t, thr=thr) for t in chosen)

    # --- Mix selection ---
    selected: list[str] = []
    selected_set: set[str] = set()

    # A0) Priority: compact evidenced bank/llm entities — highest ROI for both-tags
    priority = sorted(
        [
            s
            for s in scored
            if s.evidenced
            and s.source in ("bank", "llm")
            and 2 <= len(s.tag.split()) <= 3
        ],
        key=lambda s: -s.cosine,
    )
    for s in priority:
        if len(selected) >= n_anchors:
            break
        if _too_close(s.tag, selected, thr=0.40):
            continue
        selected.append(s.tag)
        selected_set.add(s.tag)

    # A1) Fill remaining both-slots with diverse evidenced tags
    evidenced_sorted = sorted(
        [s for s in scored if s.evidenced and s.tag not in selected_set],
        key=lambda s: (
            0 if 2 <= len(s.tag.split()) <= 3 else 1,
            -s.cosine,
            0 if s.source in ("llm", "bank") else 1,
        ),
    )

    for s in evidenced_sorted:
        if len(selected) >= n_anchors:
            break
        if _too_close(s.tag, selected, thr=0.45):
            continue
        selected.append(s.tag)
        selected_set.add(s.tag)

    # If diversity left us short, fill with next evidenced by cosine
    if len(selected) < n_anchors:
        for s in sorted([x for x in scored if x.evidenced], key=lambda x: -x.cosine):
            if s.tag in selected_set:
                continue
            selected.append(s.tag)
            selected_set.add(s.tag)
            if len(selected) >= n_anchors:
                break

    n_anchor_kept = len(selected)

    # B) Uniques: high cosine near-misses (prefer paraphrase / non-identical surface)
    unique_pool = sorted(
        [s for s in scored if s.tag not in selected_set],
        key=lambda s: (
            s.cosine,
            1 if s.source == "paraphrase" else 0,
            0 if s.evidenced else 1,
        ),
        reverse=True,
    )

    n_unique_kept = 0
    for s in unique_pool:
        if s.cosine < min_unique_cosine:
            continue
        if _too_close(s.tag, selected, thr=0.40):
            continue
        selected.append(s.tag)
        selected_set.add(s.tag)
        n_unique_kept += 1
        if len(selected) >= max_total:
            break
        if n_unique_kept >= n_uniques and len(selected) >= min_total:
            break

    # Pad if short (still respect cosine floor)
    if len(selected) < min_total:
        for s in sorted(scored, key=lambda x: -x.cosine):
            if s.tag in selected_set:
                continue
            if s.cosine < (min_unique_cosine - 0.05):
                continue
            selected.append(s.tag)
            selected_set.add(s.tag)
            if len(selected) >= min_total:
                break

    # Ensure we kept some unique surface forms when possible
    if n_unique_kept < 3:
        anchor_kept = set(selected[:n_anchor_kept])
        for s in unique_pool:
            if s.tag in selected_set or s.cosine < min_unique_cosine:
                continue
            if s.evidenced and s.tag in anchor_kept:
                continue
            selected.append(s.tag)
            selected_set.add(s.tag)
            n_unique_kept += 1
            if n_unique_kept >= 3 or len(selected) >= max_total:
                break

    return selected[:max_total]


def build_candidate_pool(
    *,
    anchors: Iterable[str],
    bank_hits: Iterable[str],
    llm_tags: Iterable[str],
    paraphrases: Iterable[str],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for t in anchors:
        out.append((t, "anchor"))
    for t in bank_hits:
        out.append((t, "bank"))
    for t in llm_tags:
        out.append((t, "llm"))
    for t in paraphrases:
        out.append((t, "paraphrase"))
    return out
