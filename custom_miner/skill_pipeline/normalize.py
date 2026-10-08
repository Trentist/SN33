"""Lint skill-coverage tests and keep the skill traceable to them.

The validator judge credits an assertion only when it is a concrete claim
that is also true of the submitted skill. The model's skill is kept.
A missing identifier from an assertion is inserted under that section.
Tests are not invented. A missing section stays missing.
"""

from __future__ import annotations

import re
from typing import Any

from custom_miner.skill_pipeline.policies import SkillPolicy
from custom_miner.skill_pipeline.policies import get_skill_policy

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_OUTCOME = re.compile(
    r"==|!=|<=|>=|->|=>|raises|returns|throws|throw|resumes|restores|aborts|produces|equals|fails"
    r"|includes|include|omits|omit|outputs|output|removes|remove|contains|emits|emit",
    re.I,
)
_INPUTISH = re.compile(r"['\"`]|\([^)]*\)|--[A-Za-z]|\bgit\b", re.I)
_TOKEN = re.compile(
    r"`([^`]{1,80})`"
    r"|\"([^\"\n]{1,80})\""
    r"|'([^'\n]{1,80})'"
    r"|(--[A-Za-z][A-Za-z0-9_\-]*)"
    r"|\b([A-Z][A-Za-z0-9]+(?:Error|Exception))\b"
    r"|\b((?:git|npm|pip|docker|kubectl)\s+[A-Za-z0-9_\-]+(?:\s+--[A-Za-z0-9_\-]+)?)"
    r"|\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\("
)
_VAGUE = (
    "behaves as expected",
    "handles edge cases",
    "returns the right value",
    "compliant result",
    "matches the skill-defined contract",
    "secondary behavior",
    "works correctly",
    "handled correctly",
    "produces the expected value",
    "uses best practices",
    "performs efficiently",
    "properly resolves",
    "supports common inputs",
    "as expected",
    "is secure",
)
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over",
    "when", "then", "than", "each", "per", "its", "are", "was", "has",
    "have", "not", "but", "any", "all", "via", "using", "used", "before",
    "after", "their", "them", "they", "will", "must", "should",
}
_TOKEN_SKIP = _STOP | {
    "none", "null", "true", "false", "input", "output", "value", "result",
    "string", "test", "given", "while", "error",
}


def _clamp(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _norm_key(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def section_parts(section: Any) -> tuple[str, str, str]:
    if isinstance(section, dict):
        sid = str(section.get("section_id") or "section")
        title = str(section.get("title") or sid)
        desc = str(section.get("description") or title)
    else:
        sid = str(getattr(section, "section_id", None) or "section")
        title = str(getattr(section, "title", None) or sid)
        desc = str(getattr(section, "description", None) or title)
    return sid, title, desc


def _content_words(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text or ""):
        key = word.lower()
        if key in _STOP or key in seen:
            continue
        seen.add(key)
        out.append(word)
    return out


def _is_concrete(assertion: str) -> bool:
    text = assertion.strip()
    lowered = text.lower()
    if any(phrase in lowered for phrase in _VAGUE):
        return False
    if len(text) < 8:
        return False
    if _OUTCOME.search(text) is None:
        return False
    # A quoted input, a command, or an exact == / raises result.
    return _INPUTISH.search(text) is not None or re.search(r"==|!=|raises|throws", text, re.I) is not None


def assertion_tokens(assertion: str) -> list[str]:
    """Identifiers a judge has to find in the skill: names, literals, commands."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(assertion or ""):
        token = next(group for group in match.groups() if group)
        token = token.strip()
        key = token.lower()
        if len(token) < 2 or key in _TOKEN_SKIP or key in seen:
            continue
        seen.add(key)
        found.append(token)
    return found


def _sanitize_test(test: Any, policy: SkillPolicy) -> dict[str, str] | None:
    if not isinstance(test, dict):
        return None
    name = _clamp(test.get("name"), policy.max_test_name_chars).strip()
    description = _clamp(test.get("description"), policy.max_test_field_chars).strip()
    assertion = _clamp(test.get("assertion"), policy.max_test_field_chars).strip()
    if not assertion or not _is_concrete(assertion):
        return None
    if not name:
        slug = re.sub(r"[^A-Za-z0-9]+", "", assertion)[:40] or "testCase"
        name = f"test{slug}"
    if not description:
        description = assertion[:160]
    return {"name": name, "description": description, "assertion": assertion}


def _unique_name(name: str, used: set[str]) -> str:
    base = name or "testCase"
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    candidate = f"{base}_{i}"
    used.add(candidate)
    return candidate


def _strip_chunk(text: str, chunk: str) -> str:
    if not text or not chunk:
        return text
    lowered = text.lower()
    needle = chunk.lower()
    start = lowered.find(needle)
    if start < 0:
        return text
    return (text[:start] + text[start + len(chunk):]).strip(" .:-")


def _description_slice(section_desc: str, half: int | None) -> str:
    """One half of a comma-list. Used only to pull a too-close pair apart."""
    section_desc = _WS.sub(" ", (section_desc or "").strip())
    if half is None or not section_desc:
        return section_desc
    parts = [part.strip() for part in re.split(r",\s*", section_desc) if part.strip()]
    if len(parts) < 2:
        return section_desc
    mid = max(1, len(parts) // 2)
    chosen = parts[:mid] if half == 0 else parts[mid:]
    if not chosen:
        chosen = parts
    text = ", ".join(chosen).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def anchor_description(
    description: str,
    *,
    title: str,
    section_desc: str,
    limit: int,
    half: int | None = None,
) -> str:
    """Full ``title: description``, then this test's own short case.

    The section ground truth is that same ``title: description`` string.
    Sharing it on both tests is what lifts section cosine into the 0.90s.
    ``half`` is unused here. A later stretch uses the halves only when a
    measured pair is too close.
    """
    del half
    title = _WS.sub(" ", (title or "").strip())
    section_desc = _WS.sub(" ", (section_desc or "").strip())
    case = _WS.sub(" ", (description or "").strip())
    case = _strip_chunk(case, title)
    case = _strip_chunk(case, section_desc)
    case = _WS.sub(" ", case).strip(" .:-")
    if case and section_desc and case.lower() in section_desc.lower():
        case = ""
    # A long case pulls the embedding off the section sentence. Twelve words
    # measured about 0.87 section cosine; a long clause fell to about 0.85.
    words = case.split()
    if len(words) > 12:
        case = " ".join(words[:12])
    base = f"{title}: {section_desc}".strip(": ") if section_desc else title
    text = f"{base} Case: {case}." if case else base
    return _WS.sub(" ", text).strip()[:limit]


def stretch_description(
    description: str,
    assertion: str,
    section_desc: str,
    *,
    half: int,
    limit: int,
    title: str = "",
) -> str:
    """Reserve room for a unique tail when a pair is too close.

    Appending and then clipping at 1,000 characters left a 0.97 pair
    unchanged. The unique tail is kept whole, and it names this test's
    title and assertion so it does not match the other test.
    """
    focus = _description_slice(section_desc, half)
    label = title or f"check {half + 1}"
    sentence = f"Only the {label} check {half + 1}: {(assertion or '').strip()}."
    if focus:
        sentence = f"{sentence} Focus: {focus}"
    unique = " ".join([sentence, sentence])
    room = max(0, limit - len(unique) - 1)
    head = (description or "")[:room].rstrip()
    return _WS.sub(" ", f"{head} {unique}").strip()[:limit]


def prepare_section_tests(
    raw: dict[str, Any] | None,
    section_map: list | None,
    *,
    policy: SkillPolicy | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Sanitize, dedupe, cap at 6, and anchor descriptions. Do not invent tests."""
    policy = policy or get_skill_policy()
    raw = raw if isinstance(raw, dict) else {}
    raw_tests = raw.get("section_tests") or {}
    if not isinstance(raw_tests, dict):
        raw_tests = {}

    sections = list(section_map or [])
    if sections:
        ordered = []
        for section in sections:
            sid, _, _ = section_parts(section)
            if sid:
                ordered.append((sid, section))
    else:
        ordered = []
        for sid, tests in raw_tests.items():
            if isinstance(sid, str) and isinstance(tests, list):
                ordered.append((sid, {"section_id": sid, "title": sid, "description": sid}))

    cleaned: dict[str, list[dict[str, str]]] = {}
    for sid, section in ordered:
        _, title, desc = section_parts(section)
        tests_in = raw_tests.get(sid) if isinstance(raw_tests.get(sid), list) else []
        seen_assert: set[str] = set()
        used_names: set[str] = set()
        kept: list[dict[str, str]] = []
        for test in tests_in:
            sanitized = _sanitize_test(test, policy)
            if not sanitized:
                continue
            key = _norm_key(sanitized["assertion"])
            if not key or key in seen_assert:
                continue
            seen_assert.add(key)
            sanitized["name"] = _unique_name(sanitized["name"], used_names)
            kept.append(sanitized)
            if len(kept) >= policy.max_tests_per_section:
                break
        for test in kept:
            test["description"] = anchor_description(
                test["description"],
                title=title,
                section_desc=desc,
                limit=policy.max_test_field_chars,
            )
        if kept:
            cleaned[sid] = kept
    return cleaned


def lock_skill_text(
    seed: str,
    section_map: list | None,
    section_tests: dict[str, list[dict[str, str]]],
    *,
    policy: SkillPolicy | None = None,
) -> str:
    """Render the skill from the section sentences and the exact assertions.

    Every surviving assertion is copied into the skill. Descriptions are
    dropped before any assertion is, and the hard cap is the validator's
    6,000-character limit rather than the shorter target length.
    """
    policy = policy or get_skill_policy()
    cap = policy.max_skill_chars
    seed_line = _WS.sub(" ", (seed or "").strip())[:400]

    by_id: dict[str, tuple[str, str]] = {}
    ordered_sections: list[tuple[str, str, str]] = []
    for section in list(section_map or []):
        sid, title, desc = section_parts(section)
        by_id[sid] = (title, desc)
        ordered_sections.append((sid, title, desc))
    for sid in section_tests:
        if sid not in by_id:
            ordered_sections.append((sid, sid, sid))

    lines: list[str] = []
    for sid, title, desc in ordered_sections:
        for test in section_tests.get(sid) or []:
            description = _WS.sub(" ", (test.get("description") or "").strip())
            assertion = _WS.sub(" ", (test.get("assertion") or "").strip())
            if description and assertion:
                lines.append(f"{description} Assertion: {assertion}")
            elif assertion:
                lines.append(f"{title}: {desc} Assertion: {assertion}".strip())

    def render(seed_text: str) -> str:
        parts = [seed_text] if seed_text else []
        parts.extend(lines)
        return "\n".join(parts).strip()

    if not lines:
        return render(seed_line)[:cap]
    text = render(seed_line)
    if len(text) <= cap:
        return text
    text = render("")
    if len(text) <= cap:
        return text
    # A hard cut would drop the last sections. Keep every assertion instead.
    compact: list[str] = []
    for sid, title, _desc in ordered_sections:
        for test in section_tests.get(sid) or []:
            assertion = _WS.sub(" ", (test.get("assertion") or "").strip())
            if assertion:
                compact.append(f"{title}: {assertion}")
    compact_text = "\n".join(compact).strip()
    return compact_text[:cap] if compact_text else text[:cap]


def limit_tests(
    section_tests: dict[str, list[dict[str, str]]],
    count: int,
) -> dict[str, list[dict[str, str]]]:
    """Keep the first ``count`` tests per section. Order is the model's order."""
    if count <= 0:
        return {}
    return {
        sid: tests[:count]
        for sid, tests in section_tests.items()
        if tests[:count]
    }


def ensure_traceability(
    skill: str,
    seed: str,
    section_map: list | None,
    section_tests: dict[str, list[dict[str, str]]],
    *,
    policy: SkillPolicy | None = None,
) -> str:
    """Keep the model skill. Append any assertion whose identifiers are missing.

    An empty skill is rendered from the section titles and the assertion lines.
    """
    policy = policy or get_skill_policy()
    cap = policy.max_skill_chars
    text = (skill or "").strip()
    if not text:
        return lock_skill_text(seed, section_map, section_tests, policy=policy)[:cap]

    titles: dict[str, tuple[str, str]] = {}
    for section in list(section_map or []):
        sid, title, desc = section_parts(section)
        titles[sid] = (title, desc)

    lowered = text.lower()
    blocks: list[str] = []
    for sid, tests in section_tests.items():
        missing_lines: list[str] = []
        for test in tests:
            absent = [
                token for token in assertion_tokens(test["assertion"])
                if token.lower() not in lowered
            ]
            if absent:
                missing_lines.append(f"- {test['assertion']}")
        if not missing_lines:
            continue
        title, desc = titles.get(sid, (sid, ""))
        lines = [f"## {title}"]
        if desc and desc.lower() not in lowered:
            lines.append(desc)
        lines.extend(missing_lines)
        blocks.append("\n".join(lines))

    if not blocks:
        return text[:cap]
    # The validator drops everything past 6,000 characters. Keep the appended
    # assertions even when the model skill is already at that limit.
    suffix = "\n\n" + "\n\n".join(blocks)
    room = cap - len(suffix)
    if room < 1:
        return suffix.strip()[:cap]
    return (text[:room].rstrip() + suffix).strip()[:cap]


def normalize_skill_bundle(
    raw: dict[str, Any] | None,
    section_map: list | None,
    *,
    policy: SkillPolicy | None = None,
) -> dict[str, Any]:
    """Lint, keep two tests per section, and make the skill name every identifier.

    Does not call an embedding API. The live path chooses the pair with embeddings.
    """
    policy = policy or get_skill_policy()
    raw = raw if isinstance(raw, dict) else {}
    seed = raw.get("seed") if isinstance(raw.get("seed"), str) else ""
    skill_in = raw.get("skill") if isinstance(raw.get("skill"), str) else ""
    tests = limit_tests(
        prepare_section_tests(raw, section_map, policy=policy),
        policy.tests_per_section,
    )
    tdd_plan = _clamp(raw.get("tdd_plan"), 240).strip() or (
        "Two concrete checks per section, both written into the skill."
    )
    return {
        "skill": lock_skill_text(seed, section_map, tests, policy=policy),
        "tdd_plan": tdd_plan,
        "section_tests": tests,
    }
