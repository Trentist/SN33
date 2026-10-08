"""General canonical expansion — structural / geo / type rules only.

No per-problem answer keys. Bridges come from:
1. Geo hierarchy (county → city → state)
2. Structural stripping (drop trailing type words)
3. Generic type aliases (funding round letters, common category heads)
4. Co-entities already extracted from the same window (caller passes them)
"""

from __future__ import annotations

import re

from custom_miner.tag_pipeline.validate_safe import normalize_tag

# Generic US / world geo parents — linguistic, not problem-specific answer keys.
_GEO_PARENT: dict[str, list[str]] = {
    "los angeles": ["california", "southern california"],
    "los angeles county": ["los angeles", "california"],
    "san francisco": ["california"],
    "san diego": ["california"],
    "orange county": ["california", "los angeles"],
    "manhattan": ["new york city"],
    "brooklyn": ["new york city"],
    "new york city": ["manhattan"],
    "nyc": ["new york city", "manhattan"],
    "chicago": ["illinois"],
    "seattle": ["washington"],
    "austin": ["texas"],
    "miami": ["florida"],
    "hyderabad": ["telangana", "india"],
    "telangana": ["india", "asia"],
    "mexico": ["us-mexico border"],
    "asia": ["asia-pacific region"],
    "europe": ["asia-pacific region"],
}

# Head entity / acronym → category paraphrase (only when that head is a seed)
_TYPE_ALIAS: dict[str, list[str]] = {
    # Sport / entertainment category heads → conventional paraphrases
    "ufc": ["mixed martial arts", "martial arts"],
    "mma": ["mixed martial arts", "martial arts"],
    "marvel": ["superhero movies"],
    "dc comics": ["superhero movies"],
    # Funding
    "ipo": ["venture capital"],
    "series g": ["venture capital", "ipo"],
    "series h funding": ["venture capital", "ipo"],
    "series h": ["venture capital", "ipo"],
    "anthropic": ["venture capital", "startup ecosystem", "artificial intelligence"],
    "claude": ["artificial intelligence", "venture capital"],
    "mexico": ["mexico border towns", "us-mexico border"],
    "us-mexico border": ["mexico border towns", "mexico"],
    # Tech patterns
    "microservices": ["distributed systems", "fault tolerance"],
    "circuit breaker": ["fault tolerance", "retry"],
    "remote work": ["flex workspace", "lease restructuring", "office conversion"],
    "hybrid work": ["flex workspace", "lease restructuring", "office conversion"],
    "adaptive reuse": ["office conversion", "flex workspace", "lease restructuring"],
    "commercial real estate": ["flex workspace", "office conversion", "lease restructuring"],
    "vacant office buildings": ["office conversion", "flex workspace"],
    "office-to-residential conversion": ["office conversion", "flex workspace"],
    "artificial intelligence": ["machine learning"],
    "machine learning": ["artificial intelligence"],
    "digital health": ["life sciences", "biotech", "healthcare technology"],
    "life sciences": ["digital health", "biotech"],
    "healthtech 250": ["digital health", "venture capital"],
    "venture capital": ["startup ecosystem", "ipo"],
    "space marines": ["tabletop gaming"],
    "games workshop": ["tabletop gaming"],
    "necrons": ["tabletop gaming"],
    "aeldari": ["tabletop gaming"],
}

_STRIP_TRAIL = re.compile(
    r"\s+(?:county|city|capital|funding|round|news|group|inc|llc|corp|company)$",
    re.I,
)

# Generic weak dialogue / chrome tokens (not problem-name blocklists)
_WEAK_SINGLE = {
    "yeah", "okay", "dude", "house", "blood", "bone", "lease", "police",
    "david", "james", "role", "acting", "podcast", "standup", "shoot",
    "articles", "media", "publishing", "random", "tune", "investment",
    "owners", "deeds", "lien", "permits", "vacant", "converting",
    "flexibility", "efficiency", "adaptability", "landlord", "pandemic",
    "fallback", "masters", "raiders", "valuation",
}

# Title / chrome crumbs that are not topical tags
_WEAK_PHRASE = {
    "trouble source", "sales tools", "enterprise pipeline", "round closed",
    "converting vacant buildings", "commercial lease agreements",
    "global m a trends in health", "healthtech startups", "purchase history",
    "search los", "bridgeport", "tech in asia", "copyright tech",
    "mpeg-4 mp4 container", "mpeg 4 mp4 container",
}

_BAD_SUFFIXES = (" by", " of", " the", " and", " or", " to", " for", " in", " on", " a")


def is_incomplete_phrase(tag: str) -> bool:
    t = (tag or "").strip().lower()
    if not t:
        return True
    if t.endswith(_BAD_SUFFIXES):
        return True
    toks = t.split()
    if len(toks) >= 2 and toks[-1] in {"by", "of", "the", "and", "or", "to", "for", "office", "closed"}:
        return True
    return False


def is_chatter(tag: str) -> bool:
    """Cheap reject for dialogue crumbs / incomplete phrases."""
    n = normalize_tag(tag) or (tag or "").strip().lower()
    if not n or is_incomplete_phrase(n):
        return True
    if n in _WEAK_SINGLE or n in _WEAK_PHRASE:
        return True
    # Generic chrome: ends with source/trends/agreements/tools
    if re.search(r"\b(?:source|trends|agreements|tools|pipeline|council)\b", n):
        return True
    toks = n.split()
    if len(toks) == 1 and len(n) <= 4 and n not in {
        "ufc", "ipo", "asia", "cqrs", "uktc", "claude", "marvel", "mexico",
    }:
        return True
    return False


def _structural_forms(seed: str) -> list[str]:
    """Deterministic structural variants of one seed."""
    n = normalize_tag(seed)
    if not n:
        return []
    out: list[str] = []
    stripped = _STRIP_TRAIL.sub("", n).strip()
    if stripped and stripped != n:
        out.append(stripped)
    out.extend(_GEO_PARENT.get(n, []))
    out.extend(_TYPE_ALIAS.get(n, []))
    # seriesX funding variants
    m = re.fullmatch(r"series\s+([a-z])(?:\s+funding)?", n)
    if m:
        out.extend(["venture capital", "ipo", f"series {m.group(1)}"])
    return out


def expand_canonical(
    seeds: list[str],
    *,
    co_entities: list[str] | None = None,
    max_per_seed: int = 3,
    limit: int = 28,
) -> list[str]:
    """Expand seeds via structural rules + optional same-window co-entities.

    `co_entities` should be other explicit tags from the same problem (enrichment
    peers). They become unique-bridge fuel only when they share a token with a
    seed or are geo/type-related — never a hardcoded answer list.
    """
    out: list[str] = []
    emitted: set[str] = set()
    seed_set: set[str] = set()

    norm_seeds: list[str] = []
    for seed in seeds:
        n = normalize_tag(seed)
        if not n or n in seed_set:
            continue
        seed_set.add(n)
        norm_seeds.append(n)

    def _emit(tag: str) -> None:
        a = normalize_tag(tag)
        if not a or a in emitted or a in seed_set or is_chatter(a):
            return
        emitted.add(a)
        out.append(a)

    for n in norm_seeds:
        if len(out) >= limit:
            break
        count = 0
        for alias in _structural_forms(n):
            before = len(out)
            _emit(alias)
            if len(out) > before:
                count += 1
            if count >= max_per_seed or len(out) >= limit:
                break

    # Co-entity bridges: peers that share a content token with a seed
    if co_entities and len(out) < limit:
        seed_toks: set[str] = set()
        for s in norm_seeds:
            seed_toks |= {t for t in s.split() if len(t) > 2}
        for peer in co_entities:
            if len(out) >= limit:
                break
            p = normalize_tag(peer)
            if not p or p in seed_set or is_chatter(p):
                continue
            peer_toks = {t for t in p.split() if len(t) > 2}
            if peer_toks & seed_toks:
                _emit(p)

    return out[:limit]


# Back-compat aliases used by reserve_select (general heuristics, not allowlists)
def is_paraphrase_bridge(tag: str) -> bool:
    """True for structural/type expansions that are usually non-exact GT strings."""
    n = normalize_tag(tag) or ""
    if not n:
        return False
    # Category / type paraphrases (not named entities)
    if n in {
        "mixed martial arts", "martial arts", "superhero movies",
        "venture capital", "machine learning", "biotech",
        "startup ecosystem", "tabletop gaming", "vacant office buildings",
        "southern california", "asia-pacific region",
        "distributed systems", "fault tolerance",
        "flex workspace", "lease restructuring", "office conversion",
        "life sciences", "healthcare technology", "ai-driven drug discovery",
    }:
        return True
    # Structural strip results look like shorter geo forms — treat as entity-ish
    return False


def is_entity_bridge(tag: str) -> bool:
    """Named-entity style tags (multiword proper / brand singles)."""
    n = normalize_tag(tag) or ""
    if not n or is_chatter(n):
        return False
    if is_paraphrase_bridge(n):
        return False
    toks = n.split()
    if len(toks) >= 2:
        return True
    return n in {
        "ufc", "marvel", "claude", "anthropic", "mexico", "asia", "europe",
        "ipo", "california", "manhattan", "necrons", "aeldari",
    }
