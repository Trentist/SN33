"""Wikidata entity linking via MediaWiki API + on-disk cache.

Inspired by Amazon ELTE / OpenTapioca: canonicalize mentions to QIDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import bittensor as bt
import requests

_API = "https://www.wikidata.org/w/api.php"
_UA = "CGP-WikiGround/1.0 (bittensor-cgp research; offline-cacheable)"
_CACHE_DIR = Path(__file__).resolve().parent / "data" / "wd_cache"
_LOCK = threading.Lock()
_SESSION: requests.Session | None = None

# Mentions that are rarely useful as WD search seeds
_SKIP = {
    "discussion", "conversation", "people", "life", "information", "opinion",
    "article", "articles", "media", "news", "topic", "podcast", "episode",
}


@dataclass
class LinkedEntity:
    mention: str
    qid: str
    label: str
    description: str = ""
    score: float = 0.0
    aliases: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)  # filled by expand


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _UA})
        _SESSION = s
    return _SESSION


def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    d = _CACHE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def _cache_get(kind: str, key: str):
    path = _cache_path(kind, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _cache_set(kind: str, key: str, value) -> None:
    path = _cache_path(kind, key)
    try:
        path.write_text(json.dumps(value, ensure_ascii=False))
    except Exception:
        pass


def _get(params: dict, *, timeout: float = 4.0, retries: int = 3) -> dict | None:
    if os.environ.get("WIKI_OFFLINE") == "1":
        return None
    params = {**params, "maxlag": 5}
    for attempt in range(retries):
        try:
            r = _session().get(_API, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or (0.8 * (attempt + 1)))
                time.sleep(min(wait, 3.0))
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            bt.logging.debug(f"wikidata api skip: {e}")
            time.sleep(0.3 * (attempt + 1))
    return None


def search_wikidata(query: str, *, limit: int = 3) -> list[dict]:
    """Return raw wbsearchentities hits for a query."""
    q = (query or "").strip()
    if not q or len(q) < 2 or q.lower() in _SKIP:
        return []
    cached = _cache_get("search", q.lower())
    if cached is not None:
        return cached
    data = _get(
        {
            "action": "wbsearchentities",
            "search": q,
            "language": "en",
            "uselang": "en",
            "limit": limit,
            "format": "json",
            "type": "item",
        }
    )
    hits = []
    if data:
        for h in data.get("search") or []:
            hits.append(
                {
                    "id": h.get("id"),
                    "label": h.get("label") or "",
                    "description": h.get("description") or "",
                    "aliases": h.get("aliases") or [],
                }
            )
    # Never cache empty/error responses — transient API failures would poison the cache
    if hits:
        _cache_set("search", q.lower(), hits)
    return hits


def _label_similarity(mention: str, label: str, description: str) -> float:
    m = re.sub(r"[^a-z0-9 ]", " ", mention.lower()).split()
    lab = re.sub(r"[^a-z0-9 ]", " ", label.lower()).split()
    if not m or not lab:
        return 0.0
    if " ".join(m) == " ".join(lab):
        return 1.0
    inter = len(set(m) & set(lab))
    union = len(set(m) | set(lab))
    j = inter / union if union else 0.0
    # Prefer short entity pages over disambiguation / surname dumps
    desc = (description or "").lower()
    penalty = 0.0
    if "disambiguation" in desc:
        penalty += 0.5
    if desc.startswith("family name") or desc.startswith("given name"):
        penalty += 0.35
    if desc.startswith("wikimedia"):
        penalty += 0.6
    return max(0.0, j - penalty)


def link_mention(mention: str, *, context: str = "") -> LinkedEntity | None:
    """Link one mention to best Wikidata entity, or None."""
    # Try original, then Title Case / UPPER for short acronyms
    variants = [mention]
    m = (mention or "").strip()
    if m and m.lower() == m and 2 <= len(m) <= 6 and " " not in m:
        variants.append(m.upper())
    if m and " " in m:
        variants.append(m.title())

    hits: list[dict] = []
    seen_ids: set[str] = set()
    for v in variants:
        for h in search_wikidata(v, limit=5):
            qid = h.get("id") or ""
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            hits.append(h)
    if not hits:
        return None
    ctx = (context or "").lower()
    best: LinkedEntity | None = None
    best_s = -1.0
    for i, h in enumerate(hits):
        qid = h.get("id") or ""
        label = h.get("label") or ""
        desc = h.get("description") or ""
        if not qid.startswith("Q"):
            continue
        s = _label_similarity(mention, label, desc)
        # Alias exact match (ufc → Ultimate Fighting Championship)
        aliases = [a.lower() for a in (h.get("aliases") or []) if isinstance(a, str)]
        if mention.lower() in aliases or mention.lower() == label.lower():
            s = max(s, 0.95)
        s += 0.15 * max(0, 3 - i) / 3.0  # rank prior
        # Context bonus: description tokens in conversation
        desc_toks = [t for t in re.findall(r"[a-z]{4,}", desc) if t not in _SKIP][:6]
        if desc_toks and ctx:
            hit = sum(1 for t in desc_toks if t in ctx) / len(desc_toks)
            s += 0.25 * hit
        # Prefer media/org entities over surname dumps when mention is a brand word
        desc_l = desc.lower()
        if "family name" in desc_l and len(mention.split()) == 1:
            s -= 0.7
        if "given name" in desc_l:
            s -= 0.5
        # Brand/media cues in description
        if any(k in desc_l for k in ("comics", "comic book", "media franchise", "promotion", "company")):
            s += 0.2
        if s > best_s:
            best_s = s
            best = LinkedEntity(
                mention=mention,
                qid=qid,
                label=label,
                description=desc,
                score=s,
                aliases=list(h.get("aliases") or []),
            )
    # Stricter accept — ambiguous single-token brands need strong alias/label match
    min_score = 0.55 if len(mention.split()) == 1 else 0.40
    if best is None or best_s < min_score:
        return None
    return best


def link_mentions(
    mentions: list,
    *,
    context: str = "",
    max_workers: int = 8,
    limit: int = 36,
) -> list[LinkedEntity]:
    """Parallel-link top mentions; dedupe by QID keeping best score."""
    seeds = []
    seen_m: set[str] = set()
    for m in mentions:
        text = getattr(m, "text", None) or str(m)
        t = text.strip().lower()
        if not t or t in seen_m or t in _SKIP:
            continue
        seen_m.add(t)
        seeds.append(text)
        if len(seeds) >= limit:
            break

    linked: list[LinkedEntity] = []
    if not seeds:
        return linked

    def _one(m: str) -> LinkedEntity | None:
        return link_mention(m, context=context)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, m): m for m in seeds}
        for fut in as_completed(futs):
            try:
                ent = fut.result(timeout=5.0)
            except Exception:
                ent = None
            if ent:
                linked.append(ent)

    by_qid: dict[str, LinkedEntity] = {}
    for ent in linked:
        prev = by_qid.get(ent.qid)
        if prev is None or ent.score > prev.score:
            by_qid[ent.qid] = ent
    out = sorted(by_qid.values(), key=lambda e: e.score, reverse=True)
    bt.logging.debug(f"wiki_link linked={len(out)}/{len(seeds)}")
    return out


def fetch_entities(qids: list[str]) -> dict[str, dict]:
    """Batch wbgetentities for claims + labels. Returns qid -> entity dict."""
    clean = [q for q in qids if q and q.startswith("Q")]
    if not clean:
        return {}
    out: dict[str, dict] = {}
    missing: list[str] = []
    for q in clean:
        cached = _cache_get("entity", q)
        if cached is not None:
            out[q] = cached
        else:
            missing.append(q)

    for i in range(0, len(missing), 40):
        chunk = missing[i : i + 40]
        data = _get(
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels|aliases|claims|descriptions",
                "languages": "en",
                "format": "json",
            },
            timeout=8.0,
        )
        entities = (data or {}).get("entities") or {}
        for qid, ent in entities.items():
            if not isinstance(ent, dict) or "missing" in ent:
                continue
            slim = {
                "id": qid,
                "labels": ent.get("labels") or {},
                "aliases": ent.get("aliases") or {},
                "descriptions": ent.get("descriptions") or {},
                "claims": {
                    pid: [
                        {
                            "mainsnak": {
                                "snaktype": snak.get("mainsnak", {}).get("snaktype"),
                                "datavalue": snak.get("mainsnak", {}).get("datavalue"),
                                "property": snak.get("mainsnak", {}).get("property"),
                            }
                        }
                        for snak in (claims or [])[:8]
                        if isinstance(snak, dict)
                    ]
                    for pid, claims in (ent.get("claims") or {}).items()
                    if pid
                    in {
                        "P31",
                        "P279",
                        "P361",
                        "P17",
                        "P131",
                        "P452",
                        "P136",
                        "P495",
                        "P106",
                        "P27",
                        "P170",
                        "P123",
                        "P127",
                        "P749",
                        "P178",
                        "P277",
                        "P1056",
                        "P101",
                    }
                },
            }
            _cache_set("entity", qid, slim)
            out[qid] = slim
        # Be polite to the public API
        if i + 40 < len(missing):
            time.sleep(0.05)
    return out
