"""10-step reserve selection for SN33 tag scoring.

General pipeline — no per-problem answer keys:
1. Split sources (conversation / enrichment)
2. Extract explicit entities + concepts (deterministic)
3. Canonical expansion (structural geo/type + co-entities)
4. LLM tags (already mined in parallel upstream)
5. Build ~30 candidate pool
6. Classify: EXACT-LIKELY / UNIQUE-BRIDGE / WEAK
7. Score exact candidates
8. Score unique-bridge candidates
9. Reserve: 5–7 exact + 3 premium unique + remainder best global
10. Output ~10–14 validate-safe tags
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import bittensor as bt

from custom_miner.tag_pipeline.anchors import extract_anchors
from custom_miner.tag_pipeline.canonical import expand_canonical
from custom_miner.tag_pipeline.canonical import is_chatter
from custom_miner.tag_pipeline.canonical import is_entity_bridge
from custom_miner.tag_pipeline.canonical import is_paraphrase_bridge
from custom_miner.tag_pipeline.tag_bank import get_tag_bank
from custom_miner.tag_pipeline.text_extract import extract_task_texts
from custom_miner.tag_pipeline.validate_safe import dedupe_near
from custom_miner.tag_pipeline.validate_safe import evidenced_in
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.tag_pipeline.validate_safe import near_dupe
from custom_miner.tag_pipeline.validate_safe import normalize_tag
from custom_miner.timing import TimeBudget

Kind = Literal["exact_likely", "unique_bridge", "weak"]

_DEDUPE = 0.40
_POOL_TARGET = 30
_N_PREMIUM_UNIQUE = 3
_OUT_MIN, _OUT_MAX = 10, 14


@dataclass
class Cand:
    tag: str
    sources: set[str] = field(default_factory=set)
    kind: Kind = "weak"
    exact_score: float = 0.0
    unique_score: float = 0.0

    @property
    def global_score(self) -> float:
        if self.kind == "exact_likely":
            return self.exact_score + 0.20
        if self.kind == "unique_bridge":
            return self.unique_score + 0.18
        return max(self.exact_score, self.unique_score) * 0.35


def _count_mentions(tag: str, text_l: str) -> int:
    if not tag or not text_l:
        return 0
    n = 0
    start = 0
    while True:
        i = text_l.find(tag, start)
        if i < 0:
            break
        n += 1
        start = i + max(len(tag), 1)
        if n >= 8:
            break
    if n:
        return n
    toks = [t for t in tag.split() if len(t) > 2]
    if len(toks) >= 2 and all(t in text_l for t in toks[:2]):
        return 1
    return 0


def _source_support(tag: str, conv_l: str, enr_l: str) -> float:
    in_c = bool(conv_l and evidenced_in(tag, conv_l))
    in_e = bool(enr_l and evidenced_in(tag, enr_l))
    if in_c and in_e:
        return 1.0
    if in_e:
        return 0.85
    if in_c:
        return 0.45
    return 0.0


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _place_boosts(enr_l: str, conv_l: str) -> list[str]:
    blob = f"{enr_l}\n{conv_l}"
    out: list[str] = []
    checks = [
        ("manhattan", "manhattan"),
        ("new york", "new york city"),
        ("nyc", "new york city"),
        ("los angeles county", "los angeles county"),
        ("los angeles", "los angeles"),
        ("california", "california"),
        ("southern california", "southern california"),
    ]
    for needle, tag in checks:
        if needle in blob:
            out.append(tag)
    return filter_validate_safe(out)


def select_reserved(
    task,
    llm_tags: list[str] | None = None,
    *,
    budget: TimeBudget | None = None,
) -> list[str]:
    """Run the 10-step reserve selector. Returns 10–14 tags."""
    texts = extract_task_texts(task)
    conversation = texts["primary"] or ""
    enrichment = texts["enrichment"] or ""
    conv_l = conversation.lower()
    enr_l = enrichment.lower()
    combined_l = (conversation + "\n" + enrichment).lower()

    def _combined_anchors():
        return filter_validate_safe(
            extract_anchors(
                conversation,
                enrichment,
                limit=22,
                primary_weight=3.0,
                enrichment_weight=5.0,
            )
        )

    def _bank():
        q = (conversation[:1500] + "\n" + enrichment[:1500]).strip()
        try:
            return filter_validate_safe(get_tag_bank().retrieve(q, k=12))
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_combined_anchors)
        fut_b = pool.submit(_bank)
        anchors = fut_a.result()
        bank_hits = fut_b.result()

    places = _place_boosts(enr_l, conv_l)
    conv_ents = [t for t in anchors if evidenced_in(t, conv_l) and not is_chatter(t)]
    enr_ents = [t for t in anchors if evidenced_in(t, enr_l) and not is_chatter(t)]
    explicit = dedupe_near(
        [t for t in (enr_ents + places + anchors) if not is_chatter(t)],
        thr=_DEDUPE,
    )

    # Structural expand + co-entities from the same window (not answer keys)
    expanded = expand_canonical(
        explicit,
        co_entities=enr_ents + places,
        max_per_seed=3,
        limit=24,
    )

    llm_raw = filter_validate_safe(
        [t for t in (llm_tags or []) if isinstance(t, str) and not is_chatter(normalize_tag(t) or "")]
    )
    expand_set = set(expanded)
    # Keep LLM tags that are evidenced or on-domain vs explicit centroid
    centroid_toks: set[str] = set()
    for t in (enr_ents + places + explicit)[:14]:
        centroid_toks |= set(t.split())
    llm_clean = dedupe_near(
        [
            t
            for t in llm_raw
            if evidenced_in(t, combined_l)
            or t in expand_set
            or _token_overlap(t, " ".join(centroid_toks)) >= 0.15
        ],
        thr=_DEDUPE,
    )

    pool_map: dict[str, Cand] = {}

    def _add(tag: str, source: str) -> None:
        n = normalize_tag(tag)
        if not n or is_chatter(n):
            return
        c = pool_map.get(n)
        if c is None:
            pool_map[n] = Cand(tag=n, sources={source})
        else:
            c.sources.add(source)

    for t in enr_ents:
        _add(t, "enrichment")
    for t in places:
        _add(t, "enrichment")
    for t in conv_ents:
        _add(t, "conversation")
    for t in anchors:
        if t not in pool_map and not is_chatter(t):
            _add(t, "conversation" if evidenced_in(t, conv_l) else "enrichment")
    for t in expanded:
        _add(t, "expand")
    for t in llm_clean:
        _add(t, "llm")
    for t in bank_hits:
        if evidenced_in(t, combined_l) or t in expand_set or _token_overlap(t, " ".join(centroid_toks)) >= 0.2:
            _add(t, "bank")

    ordered_tags = dedupe_near(explicit + expanded + llm_clean + bank_hits, thr=_DEDUPE)[:_POOL_TARGET]
    pool = [pool_map[t] for t in ordered_tags if t in pool_map]
    if len(pool) < 8:
        for t in filter_validate_safe(explicit + llm_clean + expanded):
            _add(t, "pad")
        pool = list(pool_map.values())[:_POOL_TARGET]

    # --- Classify + score (general rules) ---
    for c in pool:
        mentions_e = _count_mentions(c.tag, enr_l)
        mentions_c = _count_mentions(c.tag, conv_l)
        support = _source_support(c.tag, conv_l, enr_l)
        in_text = support > 0
        from_expand = "expand" in c.sources
        from_llm = "llm" in c.sources
        from_enr = "enrichment" in c.sources
        n_tok = len(c.tag.split())
        strong = is_entity_bridge(c.tag) or n_tok >= 2

        if is_chatter(c.tag):
            c.kind = "weak"
        elif in_text and (from_enr or mentions_e >= 1 or (strong and support >= 0.45)):
            c.kind = "exact_likely"
        elif from_expand and not in_text:
            c.kind = "unique_bridge"
        elif from_llm and not in_text and strong:
            c.kind = "unique_bridge"
        elif in_text and strong:
            c.kind = "exact_likely"
        else:
            c.kind = "weak"

        multi_src = 1.0 if len(c.sources & {"conversation", "enrichment"}) == 2 else 0.0
        c.exact_score = (
            0.30 * min(mentions_e * 2 + mentions_c, 4) / 4.0
            + 0.35 * support
            + 0.15 * multi_src
            + 0.20 * (1.0 if from_enr else 0.0)
        )
        if c.kind == "weak":
            c.exact_score *= 0.4

        overlap = _token_overlap(c.tag, " ".join(centroid_toks)) if centroid_toks else 0.0
        novelty = 0.0 if in_text else 0.40
        para_boost = 0.15 if is_paraphrase_bridge(c.tag) else 0.0
        c.unique_score = (
            0.40 * overlap
            + 0.25 * (1.0 if from_expand else 0.5 if from_llm else 0.1)
            + novelty
            + para_boost
        )
        if c.kind == "weak":
            c.unique_score *= 0.25

    exacts = sorted(
        [c for c in pool if c.kind == "exact_likely"],
        key=lambda c: c.exact_score,
        reverse=True,
    )
    uniques = sorted(
        [c for c in pool if c.kind == "unique_bridge"],
        key=lambda c: c.unique_score,
        reverse=True,
    )

    selected: list[str] = []
    selected_set: set[str] = set()

    def _take(cands: list[Cand], n: int, *, score_attr: str = "exact_score") -> int:
        added = 0
        for c in cands:
            if added >= n:
                break
            if c.tag in selected_set or is_chatter(c.tag):
                continue
            if any(near_dupe(c.tag, s, thr=_DEDUPE) for s in selected):
                continue
            selected.append(c.tag)
            selected_set.add(c.tag)
            added += 1
        return added

    _take(exacts, 6, score_attr="exact_score")
    _take(uniques, _N_PREMIUM_UNIQUE, score_attr="unique_score")

    leftovers = sorted(
        [c for c in pool if c.tag not in selected_set and not is_chatter(c.tag)],
        key=lambda c: c.global_score,
        reverse=True,
    )
    target = min(_OUT_MAX, max(_OUT_MIN, 12))
    _take(leftovers, target - len(selected), score_attr="exact_score")

    # Ensure ≥3 non-evidenced tags when possible (unique-count penalty)
    n_novel = sum(1 for t in selected if not evidenced_in(t, combined_l))
    if n_novel < 3:
        for c in uniques:
            if n_novel >= 3:
                break
            if c.tag in selected_set or evidenced_in(c.tag, combined_l):
                continue
            drop_i = next(
                (i for i, t in enumerate(selected) if evidenced_in(t, combined_l)),
                None,
            )
            if drop_i is None:
                break
            dropped = selected.pop(drop_i)
            selected_set.discard(dropped)
            selected.append(c.tag)
            selected_set.add(c.tag)
            n_novel += 1

    selected = dedupe_near(selected, thr=_DEDUPE)[:_OUT_MAX]
    if len(selected) < _OUT_MIN:
        pad = [t for t in dedupe_near(explicit + expanded + llm_clean, thr=_DEDUPE) if not is_chatter(t)]
        selected = dedupe_near(selected + pad, thr=_DEDUPE)[:_OUT_MAX]

    bt.logging.info(
        f"reserve_select enr={len(enr_ents)} conv={len(conv_ents)} expand={len(expanded)} "
        f"llm={len(llm_clean)} pool={len(pool)} out={len(selected)} tags={selected}"
    )
    if budget is not None and budget.remaining() < 0.5:
        bt.logging.debug(f"reserve_select near budget end remaining={budget.remaining():.2f}s")

    return selected[:_OUT_MAX]
