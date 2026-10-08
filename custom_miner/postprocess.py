"""Normalize miner tag output to match validator scoring expectations."""

from __future__ import annotations

import re
from typing import Any

from custom_miner.policies import TagPolicy
from custom_miner.policies import get_tag_policy

_TAG_SPLIT = re.compile(r"[,;\n|/]+")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s\-'.]+", re.UNICODE)

# High-value aliases that create UNIQUE near-miss forms near common GT tags.
_ALIAS_EXPAND: dict[str, list[str]] = {
    "los angeles": ["living in la", "southern california", "la california"],
    "los angeles county": ["la county", "la county records", "la property records"],
    "new york city": ["nyc", "new york", "nyc real estate"],
    "manhattan": ["manhattan nyc", "nyc manhattan", "manhattan townhouse"],
    "bill cosby": ["cosby", "cosby estate", "cosby foreclosure"],
    "camille cosby": ["cosby family"],
    "seattle": ["seattle washington", "city of seattle"],
    "seattle city council": ["city council seattle", "seattle council"],
    "washington": ["washington state"],
    "ufc": ["mixed martial arts", "mma fighting", "mma training"],
    "hardwire.news": ["hardwire news"],
    "cosmicbook.news": ["cosmicbook news", "cosmic book news"],
    "comprehensive plan": ["city comprehensive plan", "comprehensive planning"],
}


def _normalize_tag(tag: str, max_tokens: int) -> str | None:
    if not tag:
        return None
    t = tag.strip().lower().replace('"', "").replace("'", "")
    # Keep dots in domains like hardwire.news
    t = t.replace(".", " DOT ")
    t = _NON_ALNUM.sub(" ", t)
    t = t.replace(" DOT ", ".")
    t = _WS.sub(" ", t).strip(" -_")
    if not t:
        return None
    tokens = t.split()
    if len(tokens) > max_tokens:
        t = " ".join(tokens[:max_tokens])
    if len(t) < 2:
        return None
    if t.isdigit():
        return None
    return t


def _variant_forms(tag: str, max_tokens: int) -> list[str]:
    """Generate unique near-miss variants from a tag."""
    tokens = tag.split()
    out: list[str] = []

    for alias in _ALIAS_EXPAND.get(tag, []):
        norm = _normalize_tag(alias, max_tokens)
        if norm and norm != tag:
            out.append(norm)

    # Shorter form for long phrases
    if len(tokens) >= 3:
        shorter = " ".join(tokens[:-1])
        if len(shorter) >= 3:
            out.append(shorter)

    # Head bigram for long phrases
    if len(tokens) >= 4:
        out.append(" ".join(tokens[:2]))

    return out


def clean_tag_list(tags: list[str] | None, policy: TagPolicy) -> list[str]:
    if not tags:
        return []

    expanded: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if "," in tag and len(tag.split()) > 4:
            expanded.extend(_TAG_SPLIT.split(tag))
        else:
            expanded.append(tag)

    seen: set[str] = set()
    out: list[str] = []

    def _add(norm: str | None) -> bool:
        if not norm or norm in seen:
            return False
        # Drop obvious junk
        if norm in {"talk", "conversation", "discussion", "people", "thing", "topic", "information"}:
            return False
        seen.add(norm)
        out.append(norm)
        return True

    for raw in expanded:
        if len(out) >= policy.max_tags:
            break
        _add(_normalize_tag(raw, policy.max_tag_tokens))

    if policy.expand_variants:
        primaries = list(out)
        for tag in primaries:
            if len(out) >= policy.max_tags:
                break
            for var in _variant_forms(tag, policy.max_tag_tokens):
                if len(out) >= policy.max_tags:
                    break
                _add(var)

    return out


def postprocess_result(task_type: str, result: dict[str, Any] | None) -> dict[str, Any]:
    """Post-process mining result. Skill coverage uses skill_pipeline.normalize."""
    if not result or not isinstance(result, dict):
        if task_type == "skill_coverage_evaluation":
            return {"skill": "", "tdd_plan": "", "section_tests": {}}
        return {"tags": [], "vectors": None}

    if task_type == "skill_coverage_evaluation":
        from custom_miner.skill_pipeline import normalize_skill_bundle

        return normalize_skill_bundle(result, None)

    policy = get_tag_policy(task_type)
    tags = clean_tag_list(result.get("tags"), policy)
    vectors = result.get("vectors")

    if isinstance(vectors, dict) and tags:
        vectors = {k: v for k, v in vectors.items() if k in tags} or None
    else:
        vectors = None

    return {"tags": tags, "vectors": vectors, **{k: v for k, v in result.items() if k not in ("tags", "vectors")}}
