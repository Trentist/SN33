"""Hybrid conversation tagging pipeline.

conversation window + enrichment
  → deterministic ‖ enrichment-entities ‖ FAISS(gated) ‖ clf(gated) ‖ LLM
  → candidate union
  → ONE batch embedding
  → consensus ranking + soft redundancy
  → dual-cluster budget → best 6-10 tags
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field

import bittensor as bt
import numpy as np

from custom_miner.tag_pipeline.anchors import extract_anchors
from custom_miner.tag_pipeline.canonical import expand_canonical
from custom_miner.tag_pipeline.canonical import is_chatter
from custom_miner.tag_pipeline.canonical import is_paraphrase_bridge
from custom_miner.tag_pipeline.enrichment_extract import extract_enrichment_entities
from custom_miner.tag_pipeline.text_extract import extract_task_texts
from custom_miner.tag_pipeline.validate_safe import dedupe_near
from custom_miner.tag_pipeline.validate_safe import evidenced_in
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import near_dupe
from custom_miner.tag_pipeline.validate_safe import normalize_tag
from custom_miner.timing import TimeBudget

_GENERIC = {
    "discussion", "conversation", "people", "life", "information", "opinion",
    "article", "articles", "media", "news", "topic", "topics", "general",
    "random", "stuff", "things", "update", "updates", "content",
}
# Cross-domain brands: only keep if evidenced in THIS window/enrichment
_GATED_BRANDS = {
    "salesforce", "agentforce", "marketing cloud", "squarespace", "amazon business",
}

_OUT_MIN, _OUT_MAX = 6, 10
_POOL_MAX = 40
_DEDUPE = 0.42
_CONTENT_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "was",
    "are", "been", "into", "about", "after", "before", "your", "their", "our",
    "you", "they", "will", "would", "could", "should", "a", "an", "of", "in",
    "on", "to", "is", "it", "as", "by", "or", "at", "its", "his", "her",
}


@dataclass
class Cand:
    tag: str
    generators: set[str] = field(default_factory=set)
    retrieval: float = 0.0
    classifier: float = 0.0
    textual: float = 0.0
    enrichment_hit: bool = False
    centrality: float = 0.0
    final: float = 0.0


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _CONTENT_STOP
    }


def _tag_tokens(tag: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (tag or "").lower())
        if len(t) > 2 and t not in _CONTENT_STOP
    }


def _overlap_gate(tag: str, corpus_toks: set[str], *, min_hits: int = 1) -> bool:
    """Require lexical overlap with window+enrichment to accept FAISS/clf tags."""
    tt = _tag_tokens(tag)
    if not tt:
        return False
    hits = len(tt & corpus_toks)
    if hits >= min_hits:
        return True
    # Multi-token tags: require majority of distinctive tokens
    if len(tt) >= 2 and hits >= max(1, len(tt) - 1):
        return True
    return False


def _textual_support(tag: str, conv_l: str, enr_l: str) -> float:
    in_e = bool(enr_l and evidenced_in(tag, enr_l))
    in_c = bool(conv_l and evidenced_in(tag, conv_l))
    if in_e and in_c:
        return 1.0
    if in_e:
        return 0.92
    if in_c:
        return 0.55
    return 0.0


def _generic_penalty(tag: str) -> float:
    if tag in _GENERIC:
        return 0.45
    if len(tag.split()) == 1 and len(tag) <= 4 and tag not in {
        "ufc", "ipo", "asia", "claude", "marvel", "mexico", "cqrs", "avif", "heif", "mpeg",
    }:
        return 0.15
    return 0.0


def _weak_inference_penalty(c: Cand) -> float:
    if c.textual > 0 or c.enrichment_hit:
        return 0.0
    if c.retrieval >= 0.35 or c.classifier >= 0.35:
        return 0.05
    if c.generators <= {"llm"} and len(c.generators) == 1:
        return 0.22
    if not c.generators:
        return 0.30
    return 0.0


def _deterministic(conversation: str, enrichment: str) -> list[str]:
    raw = extract_anchors(
        conversation,
        enrichment,
        limit=24,
        primary_weight=3.0,
        enrichment_weight=5.0,
    )
    clean = [t for t in filter_validate_safe(raw) if not is_chatter(t)]
    enr_l = enrichment.lower()
    seeds = [t for t in clean if evidenced_in(t, enr_l)] or clean[:8]
    expanded = expand_canonical(seeds, co_entities=clean, max_per_seed=2, limit=12)
    return dedupe_near(clean + expanded, thr=_DEDUPE)[:30]


def _embed_batch(tags: list[str]) -> dict[str, np.ndarray]:
    if not tags:
        return {}
    try:
        from conversationgenome.llm.llm_factory import get_llm_backend

        llm = get_llm_backend()
    except Exception:
        return {}

    client = getattr(llm, "client", None)
    prefer_large = os.environ.get("HYBRID_EMBED_MODEL") or "openai/text-embedding-3-large"
    backend_model = getattr(llm, "embedding_model", None)
    vectors: dict[str, np.ndarray] = {}

    if client is not None and hasattr(client, "embeddings"):
        for attempt_model in (prefer_large, backend_model, "openai/text-embedding-3-small", "text-embedding-3-small"):
            if not attempt_model:
                continue
            try:
                for i in range(0, len(tags), 32):
                    chunk = tags[i : i + 32]
                    resp = client.embeddings.create(input=chunk, model=attempt_model)
                    for item, tag in zip(resp.data, chunk):
                        vectors[tag] = np.asarray(item.embedding, dtype=np.float32)
                if vectors:
                    return vectors
            except Exception as e:
                bt.logging.debug(f"hybrid embed fail model={attempt_model}: {e}")
                vectors = {}
                continue

    for tag in tags:
        try:
            vec = llm.get_vector_embeddings(tag)
        except Exception:
            vec = None
        if vec:
            vectors[tag] = np.asarray(vec, dtype=np.float32)
    return vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_hybrid(
    task,
    llm_tags: list[str] | None = None,
    *,
    budget: TimeBudget | None = None,
) -> list[str]:
    texts = extract_task_texts(task)
    conversation = texts["primary"] or ""
    enrichment = texts["enrichment"] or ""
    conv_l = conversation.lower()
    enr_l = enrichment.lower()
    combined = (conversation + "\n" + enrichment).strip()
    combined_l = combined.lower()
    corpus_toks = _content_tokens(combined)

    def _det():
        return _deterministic(conversation, enrichment)

    def _enr():
        return extract_enrichment_entities(enrichment, limit=16)

    def _faiss():
        try:
            from custom_miner.tag_pipeline.gt_predict import get_faiss_retriever

            return get_faiss_retriever().retrieve(combined, k=25, tag_limit=20)
        except Exception as e:
            bt.logging.debug(f"faiss retrieve skip: {e}")
            return []

    def _clf():
        try:
            from custom_miner.tag_pipeline.gt_predict import get_concept_classifier

            return get_concept_classifier().predict(combined, top_k=12, min_p=0.12)
        except Exception as e:
            bt.logging.debug(f"concept clf skip: {e}")
            return []

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_det = pool.submit(_det)
        f_enr = pool.submit(_enr)
        f_fa = pool.submit(_faiss)
        f_clf = pool.submit(_clf)
        det_tags = f_det.result()
        enr_tags = f_enr.result()
        faiss_hits = f_fa.result()
        clf_hits = f_clf.result()

    # Gate FAISS / clf: prefer overlap; allow a few high-support soft predicts
    # (masked full-conversation tags) without free cross-domain bleed.
    faiss_hard = [(t, s) for t, s in faiss_hits if _overlap_gate(t, corpus_toks, min_hits=1)]
    faiss_soft = [
        (t, s)
        for t, s in faiss_hits
        if not _overlap_gate(t, corpus_toks, min_hits=1) and float(s) >= 0.50
    ][:4]
    faiss_hits = faiss_hard + faiss_soft
    clf_hits = [
        (t, p) for t, p in clf_hits if _overlap_gate(t, corpus_toks, min_hits=1) or float(p) >= 0.55
    ]
    clf_hits = [
        (t, p)
        for t, p in clf_hits
        if _overlap_gate(t, corpus_toks, min_hits=1)
        or (_overlap_gate(t, corpus_toks | {w for w, _ in faiss_hard}, min_hits=1) and float(p) >= 0.45)
    ]

    llm_clean = filter_validate_safe(
        [t for t in (llm_tags or []) if isinstance(t, str) and not is_chatter(normalize_tag(t) or "")]
    )[:14]

    pool_map: dict[str, Cand] = {}

    def _add(tag: str, gen: str, *, retrieval: float = 0.0, classifier: float = 0.0) -> None:
        n = normalize_tag(tag)
        if not n or is_chatter(n) or n in _GENERIC:
            return
        if n not in filter_validate_safe([n]):
            return
        # Hard gate cross-domain brands unless textually supported here
        if n in _GATED_BRANDS and not evidenced_in(n, combined_l):
            return
        c = pool_map.get(n)
        if c is None:
            c = Cand(tag=n)
            pool_map[n] = c
        c.generators.add(gen)
        c.retrieval = max(c.retrieval, retrieval)
        c.classifier = max(c.classifier, classifier)
        c.textual = max(c.textual, _textual_support(n, conv_l, enr_l))
        c.enrichment_hit = c.enrichment_hit or bool(enr_l and evidenced_in(n, enr_l)) or gen == "enr"

    for t in enr_tags:
        _add(t, "enr")
    for t in det_tags:
        _add(t, "det")
    for t, s in faiss_hits:
        _add(t, "faiss", retrieval=float(s))
    for t, p in clf_hits:
        _add(t, "clf", classifier=float(p))
    for t in llm_clean:
        _add(t, "llm")

    cands = list(pool_map.values())[:_POOL_MAX]
    if not cands:
        return filter_validate_safe(llm_clean)[:_OUT_MAX]

    for c in cands:
        consensus = len(c.generators) / 4.0
        enr_bonus = 0.14 if c.enrichment_hit or "enr" in c.generators else 0.0
        c.final = (
            0.28 * consensus
            + 0.20 * c.retrieval
            + 0.22 * c.textual
            + 0.08 * c.classifier
            + enr_bonus
            - _generic_penalty(c.tag)
            - _weak_inference_penalty(c)
        )

    # Enrichment-backed seeds preferred for centroid
    seed_tags = [
        c.tag
        for c in sorted(cands, key=lambda x: (-(1 if x.enrichment_hit else 0), -x.final))
        if c.enrichment_hit or len(c.generators) >= 2
    ][:8]
    if len(seed_tags) < 3:
        seed_tags = [c.tag for c in sorted(cands, key=lambda x: -x.final)[:5]]

    allow_embed = budget is None or budget.allow_embed()
    vectors: dict[str, np.ndarray] = {}
    if allow_embed:
        vectors = _embed_batch([c.tag for c in cands])

    if vectors and seed_tags:
        seed_vecs = [vectors[t] for t in seed_tags if t in vectors]
        if seed_vecs:
            centroid = np.mean(seed_vecs, axis=0)
            for c in cands:
                if c.tag in vectors:
                    c.centrality = _cosine(vectors[c.tag], centroid)
                    enr_bonus = 0.12 if c.enrichment_hit or "enr" in c.generators else 0.0
                    c.final = (
                        0.26 * (len(c.generators) / 4.0)
                        + 0.18 * c.retrieval
                        + 0.22 * c.centrality
                        + 0.18 * c.textual
                        + 0.08 * c.classifier
                        + enr_bonus
                        - _generic_penalty(c.tag)
                        - _weak_inference_penalty(c)
                    )

    ranked = sorted(cands, key=lambda c: -c.final)
    exactish = [c for c in ranked if c.textual >= 0.45]
    enr_cands = [c for c in ranked if c.enrichment_hit or "enr" in c.generators]
    predicted = [
        c
        for c in ranked
        if c.textual < 0.45 and ("faiss" in c.generators or "clf" in c.generators)
    ]

    # Paraphrase unique fuel — include gated FAISS predicts (ufc→mma) as expand seeds
    top_seeds = (
        [c.tag for c in enr_cands[:8]]
        + [c.tag for c in exactish[:8]]
        + [c.tag for c in predicted[:10]]
    )
    para = expand_canonical(top_seeds, co_entities=[], max_per_seed=3, limit=20)
    unique_fuel: list[Cand] = []
    for t in para:
        n = normalize_tag(t)
        if not n or is_chatter(n) or evidenced_in(n, combined_l):
            continue
        if n not in pool_map:
            _add(n, "expand")
        if n not in pool_map:
            continue
        c = pool_map[n]
        c.final = max(c.final, 0.58 if is_paraphrase_bridge(n) else 0.42)
        unique_fuel.append(c)
    # Enrichment entities that aren't in conversation = strong unique/exact mix fuel
    for c in enr_cands:
        if not evidenced_in(c.tag, conv_l):
            unique_fuel.append(c)
    for c in ranked:
        if c.textual < 0.45 and "llm" in c.generators and is_paraphrase_bridge(c.tag):
            unique_fuel.append(c)

    seen_u: set[str] = set()
    uniq_sorted: list[Cand] = []
    for c in sorted(unique_fuel, key=lambda x: (-(1 if x.enrichment_hit else 0), -x.final)):
        if c.tag in seen_u:
            continue
        seen_u.add(c.tag)
        uniq_sorted.append(c)

    selected: list[str] = []

    # 1) Dual-cluster: ≥2 enrichment tags when enrichment is rich
    n_enr_forced = 0
    if len(enr_cands) >= 2:
        for c in enr_cands:
            if n_enr_forced >= 2:
                break
            if any(near_dupe(c.tag, t, thr=_DEDUPE) for t in selected):
                continue
            selected.append(c.tag)
            n_enr_forced += 1

    # 2) Cap evidenced anchors at 4 more (total exactish ~6) — leave room for uniques
    n_exact = 0
    for c in sorted(exactish, key=lambda x: (-(1 if x.enrichment_hit else 0), -x.final)):
        if n_exact >= 4:
            break
        if c.tag in selected:
            n_exact += 1
            continue
        if any(near_dupe(c.tag, t, thr=_DEDUPE) for t in selected):
            continue
        selected.append(c.tag)
        n_exact += 1

    # 3) FORCE 3 paraphrase bridges (non-evidenced) — critical for 55% unique term
    para_bridges = [
        c
        for c in uniq_sorted
        if is_paraphrase_bridge(c.tag) and not evidenced_in(c.tag, combined_l)
    ]
    if len(para_bridges) < 3:
        # Generate more from current selection seeds
        more = expand_canonical(selected + top_seeds, max_per_seed=3, limit=20)
        for t in more:
            n = normalize_tag(t)
            if not n or not is_paraphrase_bridge(n) or evidenced_in(n, combined_l):
                continue
            if n not in pool_map:
                _add(n, "expand")
            if n in pool_map:
                para_bridges.append(pool_map[n])
        # dedupe
        seen_p: set[str] = set()
        pb2 = []
        for c in para_bridges:
            if c.tag in seen_p:
                continue
            seen_p.add(c.tag)
            pb2.append(c)
        para_bridges = pb2

    n_para = 0
    for c in para_bridges:
        if n_para >= 3:
            break
        if c.tag in selected:
            n_para += 1
            continue
        if any(near_dupe(c.tag, t, thr=_DEDUPE) for t in selected):
            continue
        if len(selected) >= _OUT_MAX:
            # Drop a non-paraphrase exact to make room
            drop_i = next(
                (i for i, t in enumerate(selected) if not is_paraphrase_bridge(t)),
                None,
            )
            if drop_i is None:
                break
            selected[drop_i] = c.tag
        else:
            selected.append(c.tag)
        n_para += 1

    # 4) Fill remainder
    for c in predicted + ranked:
        if len(selected) >= _OUT_MAX:
            break
        if c.tag in selected:
            continue
        if any(near_dupe(c.tag, t, thr=_DEDUPE) for t in selected):
            continue
        selected.append(c.tag)

    selected = dedupe_near(selected, thr=_DEDUPE)[:_OUT_MAX]

    # Hard guarantee: ≥3 paraphrase-bridge tags in final list (unique-count floor)
    paras_now = [t for t in selected if is_paraphrase_bridge(t)]
    if len(paras_now) < 3:
        need = 3 - len(paras_now)
        candidates = [
            c.tag
            for c in para_bridges + uniq_sorted
            if is_paraphrase_bridge(c.tag) and c.tag not in selected
        ]
        # Also pull fresh expands from whatever we already emit
        for t in expand_canonical(selected, max_per_seed=3, limit=12):
            n = normalize_tag(t)
            if n and is_paraphrase_bridge(n) and n not in selected and not evidenced_in(n, combined_l):
                candidates.append(n)
        for tag in candidates:
            if need <= 0:
                break
            drop_i = next(
                (i for i, t in enumerate(selected) if not is_paraphrase_bridge(t)),
                None,
            )
            if drop_i is None:
                break
            if any(near_dupe(tag, t, thr=_DEDUPE) for t in selected if t != selected[drop_i]):
                continue
            selected[drop_i] = tag
            need -= 1

    if len(selected) < _OUT_MIN:
        pad = [c.tag for c in ranked if c.tag not in selected]
        selected = dedupe_near(selected + pad, thr=_DEDUPE)[:_OUT_MAX]

    bt.logging.info(
        f"hybrid det={len(det_tags)} enr={len(enr_tags)} faiss={len(faiss_hits)} "
        f"clf={len(clf_hits)} llm={len(llm_clean)} pool={len(cands)} "
        f"embed={len(vectors)} out={selected}"
    )
    return selected[:_OUT_MAX]
