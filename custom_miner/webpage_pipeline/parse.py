"""Split a webpage mining task into the page body and enrichment snippets.

After validator setup, ``window[0]`` is the page markdown (already cut to 1000
characters) and later lines are ``title\\nsnippet``. Raw JSON bundles, used in
local replay, are unpacked the same way the validator does.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PAGE_CHARS = 1000
_SNIPPET_CHARS = 1000
_STUB = re.compile(
    r"(this domain is for use in documentation|example\.com|example\.org|lorem ipsum)",
    re.I,
)


def is_stub_page(text: str) -> bool:
    """True for example.com / placeholder bodies. Ground truth ignores that chrome."""
    body = (text or "").strip()
    if not body:
        return True
    if _STUB.search(body) and len(body) < 400:
        return True
    return False


def _line_text(line: Any) -> str:
    if isinstance(line, (list, tuple)) and len(line) >= 2:
        return str(line[1] or "")
    return str(line or "")


def _snippet_from_hit(hit: dict) -> str:
    title = str(hit.get("title") or "").strip()
    snippet = str(hit.get("snippet") or "").strip()
    text = f"{title}\n{snippet}".strip()
    return text[:_SNIPPET_CHARS]


def unpack_webpage_json(text: str) -> tuple[str, list[str]]:
    """Return ``(page, snippets)`` when ``text`` is a raw webpage JSON blob."""
    raw = (text or "").strip()
    if not raw.startswith("{"):
        return "", []
    try:
        parsed = json.loads(raw)
    except Exception:
        return "", []
    if not isinstance(parsed, dict) or "website_markdown" not in parsed:
        return "", []
    page = str(parsed.get("website_markdown") or "")[:_PAGE_CHARS]
    snippets: list[str] = []
    enrichment = parsed.get("enrichment") or {}
    results = enrichment.get("search_results") if isinstance(enrichment, dict) else None
    if isinstance(results, dict):
        for hits in results.values():
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                snippet = _snippet_from_hit(hit)
                if snippet:
                    snippets.append(snippet)
    return page, snippets


def parse_webpage_task(task) -> tuple[str, list[str], bool]:
    """Return ``(page, snippets, coding)``.

    ``coding`` follows the validator: true only when ``input_categories``
    contains ``coding``.
    """
    data = getattr(getattr(task, "input", None), "data", None)
    window = list(getattr(data, "window", None) or [])
    categories = getattr(getattr(task, "input", None), "input_categories", None) or []
    coding = isinstance(categories, (list, tuple, set)) and "coding" in categories

    texts = [_line_text(line).strip() for line in window]
    texts = [text for text in texts if text]
    if not texts:
        return "", [], coding

    page, snippets = unpack_webpage_json(texts[0])
    if page or snippets:
        for extra in texts[1:]:
            extra_page, extra_snippets = unpack_webpage_json(extra)
            if extra_page or extra_snippets:
                snippets.extend(extra_snippets)
                if not page and extra_page:
                    page = extra_page
            else:
                snippets.append(extra[:_SNIPPET_CHARS])
        return page[:_PAGE_CHARS], [s[:_SNIPPET_CHARS] for s in snippets if s.strip()], coding

    page = texts[0][:_PAGE_CHARS]
    snippets = [text[:_SNIPPET_CHARS] for text in texts[1:]]
    return page, snippets, coding
