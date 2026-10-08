"""Repair truncated / messy LLM JSON for skill_bundle responses."""

from __future__ import annotations

import json
import re
from typing import Any


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    return s[start:]


def _close_open_structures(fragment: str) -> str:
    """Best-effort close truncated JSON using a bracket stack."""
    s = fragment.rstrip()
    # Trim trailing incomplete key after a comma
    s = re.sub(r",\s*\"[^\"]*$", "", s)
    s = re.sub(r",\s*$", "", s)

    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if in_string:
        s += '"'
        # After closing a truncated string, we may still be inside an object
        # that needs a closing brace before arrays/objects above it.

    # If we ended with a bare key `:` with no value, give empty string
    if re.search(r":\s*$", s):
        s += '""'

    closers = []
    for opener in reversed(stack):
        closers.append("}" if opener == "{" else "]")
    s += "".join(closers)
    return s


def repair_skill_bundle_json(text: str) -> dict[str, Any] | None:
    """Parse skill bundle JSON, repairing truncation when possible."""
    raw = _extract_json_object(text)
    if not raw:
        return None

    candidates = [raw, _close_open_structures(raw)]
    # Extra attempt: truncate to last complete test object boundary
    last_complete = raw.rfind("}")
    if last_complete > 0:
        candidates.append(_close_open_structures(raw[: last_complete + 1]))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if "skill" in parsed or "section_tests" in parsed:
            parsed.setdefault("skill", "")
            parsed.setdefault("tdd_plan", "")
            parsed.setdefault("section_tests", {})
            if not isinstance(parsed["section_tests"], dict):
                parsed["section_tests"] = {}
            return parsed
    return None
