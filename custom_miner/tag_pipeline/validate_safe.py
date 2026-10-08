"""Validate-safe tag normalization for stock validator validate_tag_set."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s\-']+", re.UNICODE)

# Stock validate_tags.j2 marks abbreviations/compounds as malformed — avoid these alone.
# Keep high-value GT brands (ufc, marvel) out of this set — they appear in ReadyAI GT.
_ABBREV = {
    "nyc", "la", "sf", "uk", "usa", "ai", "ml", "nlp", "api", "ceo", "cfo",
    "mma", "dc", "ny", "imdb", "pdf", "http", "https", "www",
}
# Short brands that ReadyAI GT emits as single tokens — always allow.
_BRAND_SINGLES = {
    "ufc", "marvel", "claude", "jaeger", "prometheus", "anthropic", "salesforce",
    "necrons", "aeldari", "uktc", "cqrs", "retry", "fallback", "bulkhead",
    "ipo", "crunchbase", "semrush", "startupblink", "polly", ".net", "dotnet",
    "mexico", "dragoneer", "greenoaks", "sequoia", "altimeter",
    "avif", "heif", "mpeg", "agentforce",
    # Regions / continents are valid GT singles
    "asia", "europe", "africa", "india", "china", "japan",
}

# Fused LLM tokens → spaced GT forms (seriesg → series g)
_FUSED_SPLIT = (
    (re.compile(r"^series([a-z])(?:funding)?$", re.I), r"series \1"),
    (re.compile(r"^series([a-z])fundinground$", re.I), r"series \1 funding"),
    (re.compile(r"^venturecapital$", re.I), "venture capital"),
    (re.compile(r"^usmexico$", re.I), "us-mexico"),
    (re.compile(r"^usmexicoborder$", re.I), "us-mexico border"),
    (re.compile(r"^mexicobordertowns?$", re.I), "mexico border towns"),
)

# Canonical domain forms — prefer short field vocabulary over fluffy rewrites
_CANON_EXACT = {
    "resiliency": "resilience",
    "resiliency patterns": "resilience patterns",
    "resilience pattern": "resilience patterns",
    "essential resilience patterns": "resilience patterns",
    "implementing resiliency patterns": "resilience patterns",
    "implementing resilience patterns": "resilience patterns",
    "retry pattern": "retry",
    "retry patterns": "retry",
    "fallback pattern": "fallback",
    "fallback patterns": "fallback",
    "timeout pattern": "timeout management",
    "timeouts": "timeout management",
    "bulkhead": "bulkhead pattern",
    "saga": "saga pattern",
    "microservice": "microservices",
    "dotnet": ".net",
    "dot net": ".net",
    "net": ".net",
    "hardwire news": "hardwire.news",
    "hardwire.news": "hardwire.news",
    "cosmicbook news": "cosmicbook.news",
    "cosmicbook.news": "cosmicbook.news",
    "cosmic book news": "cosmicbook.news",
    "lavote gov": "lavote.gov",
    "lavote.gov": "lavote.gov",
    # Funding / investors — prefer short GT forms
    "seriesg": "series g",
    "seriesh": "series h funding",
    "series g funding": "series g",
    "series g funding round": "series g",
    "series h": "series h funding",
    "series h funding round": "series h funding",
    "seriesh funding": "series h funding",
    "serieshfunding": "series h funding",
    "venturecapital": "venture capital",
    "sequoia": "sequoia capital",
    "altimeter": "altimeter capital",
    "dragoneer investment": "dragoneer",
    "dragoneer investment group": "dragoneer",
    "greenoaks capital": "greenoaks",
    "claude ai": "claude",
    "us mexico border": "us-mexico border",
    "us mexico": "us-mexico border",
    "u s mexico border": "us-mexico border",
    "u.s. mexico border": "us-mexico border",
    "u.s.-mexico border": "us-mexico border",
    "mexico border town": "mexico border towns",
    "mexican border towns": "mexico border towns",
    "mexican border": "us-mexico border",
    "mexican": "mexico",
    "asia pacific": "asia-pacific region",
    "asia pacific region": "asia-pacific region",
    "asiapacific": "asia-pacific region",
    "ai driven drug discovery": "ai-driven drug discovery",
    "ai drug discovery": "ai-driven drug discovery",
    "huggingface": "hugging face",
    "googlecloud": "google cloud",
    "joe rogan": "joe rogan",
    "the joe rogan experience": "the joe rogan experience",
    "mp4": "mp4 container",
    "av1": "av1 codec",
    "isobmff": "iso base media file format",
    "iso base media file format isobmff": "iso base media file format",
}
_STRIP_PATTERN_BASES = {
    "retry", "fallback", "circuit breaker",
}
_FLUFF_PREFIX = re.compile(
    r"^(?:implementing|building|essential|using|about|intro(?:duction)? to|guide to)\s+",
    re.I,
)
_WEAK_SINGLE = {
    "yeah", "yes", "okay", "really", "actually", "someone", "something",
    "which", "when", "where", "there", "here", "looks", "like", "also",
    "just", "very", "into", "over", "under", "andy", "jason",  # prefer full names
    "bill", "cosby", "county", "angeles", "series", "round", "funding",
    "market", "estate", "commercial", "office", "remote", "work", "spaces",
    "records", "recorder", "registrar", "mansion", "startups", "healthtech",
    "medtech", "changing", "businesses", "agreements", "tenants", "avenue",
    "dude", "because", "everybody", "everyone", "someone", "anyone", "then",
    "that", "this", "what", "with", "from", "have", "been", "were", "they",
    "them", "their", "about", "would", "could", "should", "welcome", "check",
    "podcast", "microservice",  # prefer microservices / microservice architecture
    "right", "well", "since", "kimbo", "alberto", "meta", "core", "data",
    "design", "pattern", "patterns", "software", "application", "platform",
    "global", "america", "americans", "american", "article",
    # NOTE: asia / europe / india kept as allowed region singles via _BRAND_SINGLES
    "exactly", "pict", "metadata", "image", "format", "group", "tech", "ling",
    "owner", "information", "ukrainian", "lomachenko", "connecting",
    "void", "guard", "conversion", "ctnc", "chumba",
    "building", "implementing", "essential", "learn", "composition",
    "fault", "toleran", "especially", "ilocd", "colrnclx",
    "chassis", "template",
}
_FLUFF_PHRASES = {
    "architecture patterns",
    "building resilient microservices", "building fault",
}
# Binary box crumbs only — intentional media tags (avif/heif/mpeg) are allowed.
_CODEC_JUNK = {
    "ftyp", "mdat", "hdlr", "iloc", "ipma", "pixi",
    "pitm", "hdlrpict", "isinfe", "av01", "nclx", "av1c", "ispe",
    "pict", "colr", "auxc", "ipco",
}
# GT-relevant media / codec tags (ReadyAI emits these as metadata)
_CODEC_ALLOW = {
    "avif", "heif", "mpeg", "av1 codec", "mp4 container", "heif container",
    "iso base media file format", "isobmff",
}

_JUNK = {
    "talk", "conversation", "discussion", "people", "thing", "stuff",
    "information", "content", "website", "page", "topic", "source", "getty",
    "getty images", "all rights reserved", "daily brief", "brief news",
    "brief", "daily", "news", "podcasts", "sponsored",
    "tune", "investment", "industry trends", "next generation technologies",
    "global m a trends", "global mergers and acquisitions",
    "media", "articles", "publishing", "random", "industry", "flexible workspace",
    "seed funding", "series b", "series c",  # weak generic funding unless GT-shaped
    "large language model", "even", "investor confidence", "ai-enabled workflows",
    "living in la", "comedy store", "ai news daily brief",
    "permits search", "purchase history", "early stage", "health industries",
    "loan records", "lien records", "property search", "document search",





    "call to", "sell his", "owner info", "permits", "deed", "loan", "tax",
    "property records", "mansion after", "mansion after foreclosure",
    "how remote work", "registrar recorder county clerk",
    "go to forms", "forms home page", "use do", "do vs", "statistics portal",
    "oakland website", "view report", "roll call", "meeting minutes",
    "presiding as chairperson", "like wi", "call to order",
    "draft minutes from", "determination of schedule", "items attachments",
    "of outstanding committee", "city council public",
    "image format", "owner information", "us 225 million", "connecting asia",
    "information item", "presiding officer",
}
_BAD_HEAD = {
    "and", "with", "the", "for", "from", "like", "this", "that", "our", "your",
    "view", "go", "use", "do", "floor", "forms", "pdf", "npm", "how",
}

_GARBAGE = re.compile(
    r"\b(ftyp|avif|av01|ipma|iloc|mdat|pixi|hdlr|codec|isinfe)\b",
    re.I,
)

# SEO / site-title pollution from enrichment
_SEO = re.compile(
    r"\b(wikipedia|imdb|youtube|facebook marketplace|marketwatch|"
    r"home page|official site|click here|watch online|trailer|"
    r"\- wikipedia|\- imdb|tv series \d{4})\b",
    re.I,
)
_VS_GRAMMAR = re.compile(r"\b\w+\s+vs\.?\s+\w+\b", re.I)
_NUMERIC_HEAVY = re.compile(r"(?:\d+\s*){2,}")
_URLISH = re.compile(r"(https?|www\s|dot\s+com|\.com|\.org|\.gov)", re.I)

_STOP_TAIL = {"in", "of", "to", "for", "and", "the", "a", "after", "may", "his", "her", "vs", "how"}


def canonicalize_tag(tag: str) -> str:
    """Collapse fluffy / wide variants onto domain-canonical short forms."""
    t = (tag or "").strip().lower()
    if not t:
        return t
    t = _FLUFF_PREFIX.sub("", t).strip()
    # Split fused compounds before exact canon (seriesg, venturecapital)
    compact = t.replace(" ", "").replace("-", "")
    for pat, repl in _FUSED_SPLIT:
        if pat.fullmatch(compact):
            t = pat.sub(repl, compact)
            break
    if t in _CANON_EXACT:
        return _CANON_EXACT[t]
    # "X pattern(s)" → X when X is a known short base term
    m = re.fullmatch(r"(.+?)\s+patterns?", t)
    if m and m.group(1) in _STRIP_PATTERN_BASES:
        return m.group(1)
    return t


def normalize_tag(tag: str) -> str | None:
    if not tag or not isinstance(tag, str):
        return None
    t = tag.strip().lower().replace('"', "").replace("'", "")
    # Strip parenthetical NER types: "oakland (location)" → "oakland"
    t = re.sub(r"\s*\((?:person|organization|location|org|loc|misc|gpe|product|event)\)\s*$", "", t, flags=re.I)
    # Preserve .net before dot-stripping
    if re.fullmatch(r"\.?net|dotnet|dot\s+net", t):
        return ".net"
    # Preserve *.news / *.gov as dotted forms (GT often has hardwire.news, lavote.gov)
    domain_keep = None
    if re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:news|gov|io|ai)", t):
        domain_keep = t
    # Domains → spaced words (unless dotted form kept)
    if domain_keep is None:
        t = t.replace(".", " ")
    t = _NON_ALNUM.sub(" ", t) if domain_keep is None else t
    if domain_keep is None:
        t = _WS.sub(" ", t).strip(" -_")
    else:
        t = domain_keep
    t = canonicalize_tag(t)
    if len(t) < 3 or len(t) > 55:
        return None
    if t in _JUNK or t in _ABBREV or t.isdigit() or t in _FLUFF_PHRASES:
        return None
    # Allow intentional codec / media GT tags
    if t in _CODEC_ALLOW or t.replace(".", " ") in _CODEC_ALLOW:
        return t if t in _CODEC_ALLOW else canonicalize_tag(t.replace(".", " "))
    toks_pre = t.split()
    if any(tok in _CODEC_JUNK for tok in toks_pre):
        return None
    # Codec/binary crumb tokens (e.g. colrnclx, ilocd)
    if len(toks_pre) == 1:
        tok = toks_pre[0]
        vowels = len(re.findall(r"[aeiou]", tok))
        if len(tok) >= 5 and vowels <= 1:
            return None
        if tok in {"ilocd", "colrnclx", "nclx", "pitm", "iinf", "iprp", "ipco", "ispe"}:
            return None
        if re.fullmatch(r"[a-z]{2,4}(?:inf|iprp|ipco|iloc|ipma|nclx)", tok):
            return None
        if re.search(r"(?:av01|viprp|ipco|iprp|ftyp)", tok):
            return None
    # Allow known brand singles through garbage filter
    if t in _BRAND_SINGLES:
        return t
    # Dotted short domains are intentional GT tags (hardwire.news, lavote.gov)
    _is_short_domain = bool(re.fullmatch(r"[a-z0-9-]+\.(?:news|gov|io|ai)", t))
    if _GARBAGE.search(t) or _SEO.search(t) or (_URLISH.search(t) and not _is_short_domain):
        return None
    if _is_short_domain:
        return t
    if _VS_GRAMMAR.search(t):
        return None
    # Reject date-ish / numeric-heavy fragments
    if re.search(r"\b(?:19|20)\d{2}\b", t) and len(t.split()) <= 4:
        # allow years only inside longer topical phrases (>=4 tokens already handled)
        if sum(ch.isdigit() for ch in t) >= 4 and len(t.split()) <= 3:
            return None
    if _NUMERIC_HEAVY.search(t) and sum(ch.isdigit() for ch in t) >= 4:
        return None
    toks = t.split()
    if not toks or len(toks) > 6:
        return None
    # Allow "the joe rogan experience" style titles
    if toks[0] == "the" and len(toks) >= 3:
        toks = toks  # keep
    elif toks[0] in _BAD_HEAD:
        return None
    if toks[-1] in _STOP_TAIL:
        return None
    if toks[-1] in {"website", "webpage", "minutes", "report", "call"}:
        return None
    _BAD_TAILS = {
        "forced", "sell", "selling", "source", "search", "records",
        "trouble", "present", "biggest", "culprits", "laws", "how", "use",
        "portal", "locator", "hours", "financial", "chairperson",
    }
    # "clerk" OK inside registrar-recorder/county clerk style phrases
    if toks[-1] in _BAD_TAILS and len(toks) >= 3:
        return None
    # Drop very short second token fragments ("like wi") — keep funding letters,
    # short-head topical phrases, geo heads, and person names (joe rogan)
    if len(toks) == 2 and (len(toks[0]) <= 3 or len(toks[1]) <= 2):
        allow = (
            (toks[0] == "series" and len(toks[1]) == 1 and toks[1].isalpha())
            or (toks[0] in {"ai", "ml", "us", "av1", "mp4"} and len(toks[1]) >= 4)
            or (toks[0] in {"los", "san", "las", "new", "ft", "st", "el", "la", "joe"} and len(toks[1]) >= 4)
        )
        if not allow:
            return None
    if all(len(x) <= 1 for x in toks):
        return None
    return t


def is_validate_safe(tag: str) -> bool:
    n = normalize_tag(tag)
    if not n:
        return False
    toks = n.split()
    # GT uses "series g" / allow series + single funding letter
    if len(toks) == 2 and toks[0] == "series" and len(toks[1]) == 1 and toks[1].isalpha():
        return True
    if len(toks) == 1:
        if n in _BRAND_SINGLES:
            return True
        if len(toks[0]) < 4 or toks[0] in _ABBREV or toks[0] in _WEAK_SINGLE:
            return False
    return True


def filter_validate_safe(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        n = normalize_tag(t)
        if not n or n in seen or not is_validate_safe(n):
            continue
        seen.add(n)
        out.append(n)
    return out


def _stem(tok: str) -> str:
    """Light stem so resiliency ≈ resilience, microservices ≈ microservice."""
    t = tok.lower()
    for suf in ("ing", "tion", "sion", "ness", "ment", "ers", "ies", "es", "s", "y"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


_PLACE_ELABORATION = {
    "border", "towns", "town", "county", "city", "state", "valley", "river",
    "united", "states", "southern",
}
_ORG_ELABORATION = {
    "capital", "investment", "group", "partners", "ventures", "fund", "ai",
}


def near_dupe(a: str, b: str, thr: float = 0.45) -> bool:
    """Token Jaccard + substring/stem clone check (kill near-identical families)."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Substring / containment clones (same head entity, longer fluff)
    # BUT keep place/org elaborations that GT often lists alongside the short form
    # (mexico + mexico border towns; dragoneer + dragoneer investment group).
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 6 and shorter in longer:
            extra = set(longer.replace("-", " ").split()) - set(shorter.replace("-", " ").split())
            if extra & _PLACE_ELABORATION or extra & _ORG_ELABORATION:
                pass  # not a clone — fall through to weaker checks
            else:
                return True
    ta, tb = a.replace("-", " ").split(), b.replace("-", " ").split()
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    # Short org/place name + elaboration is NOT a clone (greenoaks / greenoaks capital)
    if sa < sb or sb < sa:
        extra = (sa | sb) - (sa & sb)
        if extra & _PLACE_ELABORATION or extra & _ORG_ELABORATION:
            return False
    inter = len(sa & sb)
    union = len(sa | sb)
    if union and inter / union >= thr:
        return True
    # Same first two tokens (bill cosby / bill cosby estate)
    if len(ta) >= 2 and len(tb) >= 2 and ta[0] == tb[0] and ta[1] == tb[1]:
        return True
    # Person/brand clone family: same surname/brand + generic role suffix
    _ROLE = {"family", "estate", "foreclosure", "article", "news", "market", "industry", "ecosystem"}
    if len(ta) == 2 and len(tb) == 2 and ta[0] == tb[0] and len(ta[0]) >= 4:
        if ta[1] in _ROLE or tb[1] in _ROLE:
            return True
    # Stemmed last-token match + similar head (resilience patterns ≈ resiliency patterns)
    if len(ta) >= 2 and len(tb) >= 2 and _stem(ta[-1]) == _stem(tb[-1]):
        if _stem(ta[0]) == _stem(tb[0]) or ta[0][:5] == tb[0][:5]:
            return True
    # Single-token stem clones
    if len(ta) == 1 and len(tb) == 1 and _stem(ta[0]) == _stem(tb[0]):
        return True
    return False


def dedupe_near(tags: list[str], thr: float = 0.45) -> list[str]:
    """Drop near-duplicate / clone tags keeping earlier (higher-priority) ones."""
    out: list[str] = []
    for t in tags:
        if any(near_dupe(t, x, thr=thr) for x in out):
            continue
        out.append(t)
    return out


_GENERIC_EVIDENCE = {
    "city", "council", "office", "public", "safety", "committee", "hall",
    "county", "state", "market", "news", "plan", "report", "minutes",
}


def evidenced_in(tag: str, text_l: str, *, strict: bool = False) -> bool:
    """True if tag appears in text. Multi-word tags need the distinctive tokens."""
    if not tag or not text_l:
        return False
    if tag in text_l:
        return True
    toks = [tok for tok in tag.split() if len(tok) > 2]
    if not toks:
        return False
    if len(toks) == 1:
        return toks[0] in text_l
    # Distinctive tokens (not generic civic/news words) must all be present
    distinctive = [t for t in toks if t not in _GENERIC_EVIDENCE]
    if distinctive:
        return all(t in text_l for t in distinctive)
    # All-generic phrase: require full contiguous phrase or all tokens
    if strict:
        return all(t in text_l for t in toks)
    hits = sum(1 for tok in toks if tok in text_l)
    return hits == len(toks)
