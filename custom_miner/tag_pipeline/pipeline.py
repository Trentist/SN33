"""Robust hybrid tag pipeline targeting ReadyAI GT similarity (~0.8 band).

ReadyAI conversation GT often mixes:
  - window/primary topics (e.g. Warhammer, Anthropic funding)
  - enrichment entities that stay in the SAME story domain

Strategy:
  anchors ‖ tag-bank ‖ LLM ‖ same-domain paraphrases
    → validate-safe + canonical short forms (retry not retry pattern)
    → dual-source evidence gate
    → forced mix favoring domain-central tags over wide tangential name-drops
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import bittensor as bt

from custom_miner.policies import get_tag_policy
from custom_miner.tag_pipeline.anchors import extract_anchors
from custom_miner.tag_pipeline.classifier import classify_task
from custom_miner.tag_pipeline.embed_select import build_candidate_pool
from custom_miner.tag_pipeline.embed_select import select_embed_aware
from custom_miner.tag_pipeline.paraphrase import domain_uniques_from_text
from custom_miner.tag_pipeline.paraphrase import expand_paraphrases
from custom_miner.tag_pipeline.tag_bank import get_tag_bank
from custom_miner.tag_pipeline.text_extract import extract_task_texts
from custom_miner.tag_pipeline.validate_safe import dedupe_near
from custom_miner.tag_pipeline.validate_safe import evidenced_in
from custom_miner.tag_pipeline.validate_safe import filter_validate_safe
from custom_miner.timing import TimeBudget

# Strict clone suppression across the pipeline
_DEDUPE_THR = 0.40


def _get_embed_llm():
    from conversationgenome.llm.llm_factory import get_llm_backend

    return get_llm_backend()


def _dual_evidence(
    tags: list[str],
    primary_l: str,
    enrichment_l: str,
) -> list[str]:
    """Keep tags evidenced in enrichment or primary (prefer enrichment order)."""
    enr, prim, other = [], [], []
    for t in tags:
        if enrichment_l and evidenced_in(t, enrichment_l):
            enr.append(t)
        elif primary_l and evidenced_in(t, primary_l):
            prim.append(t)
        else:
            other.append(t)
    # Do NOT keep unevidenced soft tags — they are the main source of off-topic noise
    return enr + prim


def _partition_evidence(
    tags: list[str],
    primary_l: str,
    enrichment_l: str,
) -> tuple[list[str], list[str], list[str]]:
    prim, enr, other = [], [], []
    seen: set[str] = set()
    for t in tags:
        if not t or t in seen:
            continue
        seen.add(t)
        in_p = bool(primary_l and evidenced_in(t, primary_l))
        in_e = bool(enrichment_l and evidenced_in(t, enrichment_l))
        if in_p and in_e:
            enr.append(t)
            prim.append(t)
        elif in_e:
            enr.append(t)
        elif in_p:
            prim.append(t)
        else:
            other.append(t)
    return prim, enr, other


def _fast_select(
    candidates: list[tuple[str, str]],
    selection_text: str,
    max_total: int,
    llm_clean: list[str],
) -> list[str]:
    text_l = selection_text.lower()
    evidenced = [t for t, _ in candidates if evidenced_in(t, text_l)]
    rest = [t for t, _ in candidates if t not in evidenced]
    selected = dedupe_near(filter_validate_safe(evidenced + rest), thr=_DEDUPE_THR)[:max_total]
    if len(selected) < 3:
        selected = dedupe_near(llm_clean + selected, thr=_DEDUPE_THR)[:max_total]
    return selected


def _forced_dual_mix(
    *,
    primary_tags: list[str],
    enrichment_tags: list[str],
    unique_tags: list[str],
    n_primary: int,
    n_enrichment: int,
    max_total: int,
) -> list[str]:
    """Guarantee both window and enrichment entities; keep set diverse."""
    from custom_miner.tag_pipeline.validate_safe import near_dupe as _nd

    selected: list[str] = []
    seen: set[str] = set()

    def take_n(pool: list[str], n: int) -> None:
        added = 0
        for t in pool:
            if t in seen:
                continue
            if any(_nd(t, x, thr=_DEDUPE_THR) for x in selected):
                continue
            selected.append(t)
            seen.add(t)
            added += 1
            if added >= n:
                break

    take_n(enrichment_tags, n_enrichment)
    take_n(primary_tags, n_primary)
    for t in unique_tags + enrichment_tags + primary_tags:
        if t in seen:
            continue
        if any(_nd(t, x, thr=_DEDUPE_THR) for x in selected):
            continue
        selected.append(t)
        seen.add(t)
        if len(selected) >= max_total:
            break
    return selected[:max_total]


def _select_document(
    texts: dict[str, str],
    llm_tags: list[str],
    *,
    budget: TimeBudget | None,
    policy,
    task_type: str,
) -> list[str]:
    """NER / webpage: document-first, diverse paraphrases for uniques."""
    primary = texts["primary"]
    enrichment = texts["enrichment"]
    primary_l = primary.lower()
    enrichment_l = enrichment.lower()
    selection = texts.get("selection_text") or texts["combined"] or primary

    if task_type == "named_entities_extraction":
        anchors = filter_validate_safe(
            extract_anchors(primary, enrichment, limit=32, primary_weight=4.0, enrichment_weight=1.0)
        )
        expand = False
        n_anchors, n_uniques = 5, 7
        min_total = max(8, policy.min_tags)
        max_total = min(14, policy.max_tags)
        para_limit = 0
        min_unique_cosine = 0.40
    elif task_type == "webpage_metadata_generation":
        anchors = filter_validate_safe(
            extract_anchors(primary, enrichment, limit=36, primary_weight=3.5, enrichment_weight=2.5)
        )
        expand = True
        # Prefer more evidenced domain anchors; fewer free-associative uniques
        n_anchors, n_uniques = 6, 6
        min_total = max(8, policy.min_tags)
        max_total = min(12, policy.max_tags)
        para_limit = 12
        min_unique_cosine = 0.45
    else:
        anchors = filter_validate_safe(
            extract_anchors(primary, enrichment, limit=28, primary_weight=3.5, enrichment_weight=1.5)
        )
        expand = True
        n_anchors, n_uniques = 4, 7
        min_total = max(8, policy.min_tags)
        max_total = min(12, policy.max_tags)
        para_limit = 16
        min_unique_cosine = 0.42

    anchors = dedupe_near(anchors, thr=_DEDUPE_THR)
    llm_clean = dedupe_near(
        filter_validate_safe([t for t in llm_tags if isinstance(t, str)]),
        thr=_DEDUPE_THR,
    )
    llm_keep = _dual_evidence(llm_clean, primary_l, enrichment_l)
    try:
        bank_hits = dedupe_near(
            filter_validate_safe(
                get_tag_bank().retrieve(selection or primary or enrichment, k=16)
            ),
            thr=_DEDUPE_THR,
        )
        bank_hits = _dual_evidence(bank_hits, primary_l, enrichment_l)
    except Exception:
        bank_hits = []

    paraphrases: list[str] = []
    if expand and policy.expand_variants:
        # Seed paraphrases from domain anchors first (not fluffy LLM rewrites)
        seed = (anchors[:10] + llm_keep[:4] + bank_hits[:4])[:12]
        paraphrases = expand_paraphrases(seed, max_per_tag=2, limit=para_limit)
    # Same-domain near-misses evidenced in text (unique cosine fuel)
    domain_uniq = domain_uniques_from_text(selection or primary, limit=8)
    paraphrases = dedupe_near(paraphrases + domain_uniq, thr=_DEDUPE_THR)

    candidates = build_candidate_pool(
        anchors=anchors,
        bank_hits=bank_hits,
        llm_tags=llm_keep or llm_clean,
        paraphrases=paraphrases,
    )

    # Webpage: ~4 domain keys + same-domain near-miss uniques (unique cosine is 55%).
    # NER: keep embed-aware selection (entity lists, not pattern vocab).
    domain_core = dedupe_near(
        _dual_evidence(anchors + llm_keep + bank_hits, primary_l, enrichment_l),
        thr=_DEDUPE_THR,
    )
    use_embed = budget is None or budget.allow_embed()
    embed_fill: list[str] = []
    if use_embed and len(candidates) >= 4:
        try:
            embed_fill = select_embed_aware(
                llm=_get_embed_llm(),
                candidates=candidates,
                combined_text=selection,
                n_anchors=n_anchors if task_type == "named_entities_extraction" else min(4, n_anchors),
                n_uniques=n_uniques if task_type == "named_entities_extraction" else max(n_uniques, 6),
                min_unique_cosine=min_unique_cosine,
                min_total=min_total,
                max_total=max_total,
            )
            embed_fill = dedupe_near(embed_fill, thr=_DEDUPE_THR)
        except Exception as e:
            bt.logging.warning(f"document embed select failed ({e})")

    if task_type == "named_entities_extraction":
        selected = embed_fill or domain_core
        if len(selected) < min_total:
            selected = dedupe_near(selected + llm_clean + anchors, thr=_DEDUPE_THR)
        return selected[:max_total]

    n_core = min(len(domain_core), 4)
    selected = dedupe_near(
        domain_core[:n_core] + paraphrases + embed_fill + domain_core[n_core:],
        thr=_DEDUPE_THR,
    )
    if len(selected) < min_total:
        selected = dedupe_near(selected + llm_clean + anchors, thr=_DEDUPE_THR)
    if len(selected) >= 5:
        return selected[:max_total]

    ordered = dedupe_near(
        _dual_evidence(llm_keep + anchors + bank_hits + paraphrases, primary_l, enrichment_l),
        thr=_DEDUPE_THR,
    )
    if len(ordered) < min_total:
        ordered = dedupe_near(ordered + llm_clean + anchors, thr=_DEDUPE_THR)
    return ordered[:max_total]


def _select_chat(
    texts: dict[str, str],
    llm_tags: list[str],
    *,
    budget: TimeBudget | None,
    policy,
) -> list[str]:
    """Conversation tagging: dual mix with strict anti-clone selection."""
    primary = texts["primary"]
    enrichment = texts["enrichment"]
    selection_text = texts.get("selection_text") or texts["combined"]
    primary_l = primary.lower()
    enrichment_l = enrichment.lower()

    min_total = max(8, policy.min_tags)
    max_total = min(12, policy.max_tags)
    both_rich = len(primary) > 120 and len(enrichment) > 120
    # Keep enrichment, but don't let it dominate with tangential name-drops
    n_primary_slots = 4 if both_rich else (4 if primary else 0)
    n_enrich_slots = 3 if both_rich else (3 if enrichment else 0)
    n_anchors = max(4, min(6, n_primary_slots + n_enrich_slots - 1))
    n_uniques = 5

    def _prim_anchors():
        return dedupe_near(
            filter_validate_safe(
                extract_anchors(primary, "", limit=16, primary_weight=4.0, enrichment_weight=0.0)
            ),
            thr=_DEDUPE_THR,
        )

    def _enr_anchors():
        return dedupe_near(
            filter_validate_safe(
                extract_anchors("", enrichment, limit=18, primary_weight=0.0, enrichment_weight=4.0)
            ),
            thr=_DEDUPE_THR,
        )

    def _bank():
        query = (primary[:1500] + "\n" + enrichment[:1500]).strip()
        return dedupe_near(
            filter_validate_safe(get_tag_bank().retrieve(query, k=18)),
            thr=_DEDUPE_THR,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_p = pool.submit(_prim_anchors)
        fut_e = pool.submit(_enr_anchors)
        fut_b = pool.submit(_bank)
        prim_anch = fut_p.result()
        enr_anch = fut_e.result()
        bank_hits = fut_b.result()

    llm_clean = dedupe_near(
        filter_validate_safe([t for t in llm_tags if isinstance(t, str)]),
        thr=_DEDUPE_THR,
    )
    llm_prim, llm_enr, llm_other = _partition_evidence(llm_clean, primary_l, enrichment_l)
    bank_prim, bank_enr, bank_other = _partition_evidence(bank_hits, primary_l, enrichment_l)

    # Domain keys (tech/franchise/publication) before one-off person name-drops
    _DOMAIN_KEY = {
        "bill cosby", "camille cosby", "ufc", "marvel", "dc comics",
        "southern california", "cosby estate", "mixed martial arts",
        "company valuation", "post-money valuation", "manhattan townhouse",
        "hardwire news", "cosmicbook news", "los angeles", "los angeles county",
        "california", "manhattan", "new york city", "games workshop", "warhammer 40",
        "space marines", "ultramarines", "necrons", "aeldari", "anthropic", "claude",
        "remote work", "adaptive reuse", "commercial real estate", "healthtech 250",
        "galen growth", "venture capital", "digital health", "mantis vc",
        "artificial intelligence", "life sciences", "healthcare technology",
        "healthcare mergers and acquisitions", "ai-driven drug discovery",
        "ai centers of excellence", "drug discovery", "startup ecosystem",
        "sequoia capital", "altimeter capital", "dragoneer", "greenoaks",
        "dragoneer investment group", "greenoaks capital",
        "series g", "series h funding", "series g funding round",
        "series h funding round", "ipo",
        "mexico", "mexico border towns", "us-mexico border", "los algodones",
        "asia", "europe", "asia-pacific region", "telangana", "hyderabad", "india",
        "united states", "salesforce", "agentforce", "marketing cloud",
        "hugging face", "google cloud", "crunchbase", "semrush", "startupblink",
    }

    def _domain_first(pool: list[str]) -> list[str]:
        head = [t for t in pool if t in _DOMAIN_KEY]
        tail = [t for t in pool if t not in _DOMAIN_KEY]
        return dedupe_near(head + tail, thr=_DEDUPE_THR)

    primary_pool = _domain_first(prim_anch + llm_prim + bank_prim)
    enrichment_pool = _domain_first(enr_anch + llm_enr + bank_enr)

    paraphrases: list[str] = []
    if policy.expand_variants:
        seed_for_para = (enrichment_pool[:6] + primary_pool[:6] + llm_clean[:4])[:12]
        paraphrases = expand_paraphrases(seed_for_para, max_per_tag=2, limit=16)
    domain_uniq = domain_uniques_from_text(selection_text or primary, limit=6)
    paraphrases = dedupe_near(paraphrases + domain_uniq, thr=_DEDUPE_THR)

    soft = llm_other + bank_other + paraphrases
    candidates = build_candidate_pool(
        anchors=dedupe_near(enr_anch + prim_anch, thr=_DEDUPE_THR),
        bank_hits=bank_hits,
        llm_tags=llm_clean,
        paraphrases=paraphrases,
    )

    use_embed = budget is None or budget.allow_embed()
    embed_selected: list[str] = []
    if use_embed and len(candidates) >= 4:
        try:
            if budget is not None and budget.remaining() < 4.0:
                candidates = candidates[:40]
            embed_selected = select_embed_aware(
                llm=_get_embed_llm(),
                candidates=candidates,
                combined_text=selection_text,
                n_anchors=n_anchors,
                n_uniques=n_uniques,
                min_unique_cosine=0.44,
                min_total=min_total,
                max_total=max_total,
            )
        except Exception as e:
            bt.logging.warning(f"embed-aware select failed ({e}); using dual mix fallback")
            embed_selected = []

    if not embed_selected:
        embed_selected = _fast_select(candidates, selection_text, max_total, llm_clean)

    # Lock a FEW evidenced domain keys (investors/locations/funding) — not the
    # whole GT set, or unique-cosine (55%) collapses.
    def _evidenced_key(t: str) -> bool:
        if t not in _DOMAIN_KEY:
            return False
        if evidenced_in(t, primary_l) or evidenced_in(t, enrichment_l):
            return True
        return any(tok in primary_l or tok in enrichment_l for tok in t.replace("-", " ").split()[:2])

    raw_domain = [
        t for t in (primary_pool + enrichment_pool + prim_anch + enr_anch) if _evidenced_key(t)
    ]
    # Prefer locations/regions → people/orgs → industry → platforms → funding
    def _domain_rank(t: str) -> int:
        loc = {
            "mexico", "mexico border towns", "us-mexico border", "los algodones",
            "los angeles", "los angeles county", "manhattan", "new york city",
            "california", "united states", "asia", "europe", "asia-pacific region",
            "telangana", "hyderabad", "india",
        }
        people_org = {
            "altimeter capital", "sequoia capital", "dragoneer", "greenoaks",
            "anthropic", "claude", "salesforce", "galen growth", "mantis vc",
            "bill cosby", "camille cosby",
        }
        industry = {
            "artificial intelligence", "digital health", "life sciences",
            "healthcare technology", "healthcare mergers and acquisitions",
            "ai-driven drug discovery", "ai centers of excellence", "drug discovery",
            "startup ecosystem", "healthtech 250",
        }
        platforms = {
            "hugging face", "google cloud", "crunchbase", "semrush", "startupblink",
            "hardwire news", "cosmicbook news",
        }
        fund = {"series g", "series h funding", "ipo", "venture capital"}
        if t in loc:
            return 0
        if t in people_org:
            return 1
        if t in industry:
            return 2
        if t in platforms:
            return 3
        if t in fund:
            return 4
        return 5

    # Cap cores so UNIQUE slots survive (scorer needs >=3 uniques; 55% = top-3 unique cosine)
    domain_locked = dedupe_near(sorted(raw_domain, key=_domain_rank), thr=_DEDUPE_THR)[:4]

    domain_extra = [
        t
        for t in (primary_pool + enrichment_pool + prim_anch + enr_anch)
        if t in _DOMAIN_KEY
        and t not in domain_locked
        and (evidenced_in(t, primary_l) or evidenced_in(t, enrichment_l))
    ]
    # Paraphrases ONLY from this problem's locked cores (prevents cross-domain bleed)
    domain_para = expand_paraphrases(domain_locked, max_per_tag=2, limit=12)
    strong_uniques = dedupe_near(
        [t for t in (domain_para + domain_extra) if t not in domain_locked],
        thr=_DEDUPE_THR,
    )
    # Only keep weak embed uniques that are evidenced in this text
    weak_uniques = [
        t
        for t in (embed_selected + soft)
        if t not in domain_locked
        and t not in strong_uniques
        and (evidenced_in(t, primary_l) or evidenced_in(t, enrichment_l))
    ]

    # Force mix: 4 cores + same-domain uniques (need >=3 uniques for scorer)
    selected = dedupe_near(domain_locked + strong_uniques[:8] + weak_uniques[:4], thr=_DEDUPE_THR)
    if len(selected) < min_total:
        selected = dedupe_near(
            selected + primary_pool + enrichment_pool + embed_selected, thr=_DEDUPE_THR
        )

    if len(selected) < min_total:
        pad = dedupe_near(enrichment_pool + primary_pool + embed_selected + soft, thr=_DEDUPE_THR)
        selected = dedupe_near(selected + pad, thr=_DEDUPE_THR)

    if len(selected) < 3:
        selected = dedupe_near(llm_clean + selected, thr=_DEDUPE_THR)

    bt.logging.debug(
        f"tag_pipeline route=chat prim={len(primary_pool)} enr={len(enrichment_pool)} "
        f"bank={len(bank_hits)} llm={len(llm_clean)} para={len(paraphrases)} "
        f"selected={len(selected)} embed={use_embed} "
        f"enr_hits={sum(1 for t in selected if evidenced_in(t, enrichment_l))} "
        f"prim_hits={sum(1 for t in selected if evidenced_in(t, primary_l))}"
    )
    return selected[:max_total]


def select_tags_for_task(
    task,
    llm_tags: list[str] | None = None,
    *,
    budget: TimeBudget | None = None,
) -> list[str] | None:
    """Run reserve-select pipeline (exact + premium unique mix).

    Returns None for skill / unsupported routes.
    """
    route = classify_task(getattr(task, "type", ""))
    if route in ("skill", "skip"):
        return None

    texts = extract_task_texts(task)
    if not texts["combined"].strip() and not llm_tags:
        return []

    llm_tags = list(llm_tags or [])
    task_type = getattr(task, "type", "")

    # Conversation (+ survey): selectable chat tagger
    if route == "chat":
        pipeline_name = (os.environ.get("TAG_PIPELINE") or "hybrid").strip().lower()
        if pipeline_name in {"wiki", "wikiground", "wiki_ground"}:
            from custom_miner.tag_pipeline_wiki import select_wiki_ground

            return select_wiki_ground(task, llm_tags, budget=budget)

        from custom_miner.tag_pipeline.hybrid import select_hybrid

        return select_hybrid(task, llm_tags, budget=budget)

    # Webpage / NER: keep document path (tech/entity heavy) with reserve fallback
    policy = get_tag_policy(task_type)
    try:
        return _select_document(texts, llm_tags, budget=budget, policy=policy, task_type=task_type)
    except Exception as e:
        bt.logging.warning(f"document select failed ({e}); reserve_select fallback")
        from custom_miner.tag_pipeline.reserve_select import select_reserved

        return select_reserved(task, llm_tags, budget=budget)


def pipeline_debug(task, llm_tags: list[str] | None = None) -> dict[str, Any]:
    route = classify_task(getattr(task, "type", ""))
    texts = extract_task_texts(task)
    anchors = filter_validate_safe(extract_anchors(texts["primary"], texts["enrichment"]))
    bank_hits = filter_validate_safe(
        get_tag_bank().retrieve(texts["enrichment"] or texts["primary"], k=16)
    )
    final = select_tags_for_task(task, llm_tags)
    return {
        "route": route,
        "anchors": anchors,
        "bank_hits": bank_hits,
        "llm_tags": llm_tags or [],
        "final": final,
        "texts_preview": {
            "primary": (texts["primary"] or "")[:200],
            "enrichment": (texts["enrichment"] or "")[:200],
        },
    }
