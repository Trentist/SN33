"""Split a named-entity task into the transcript and enrichment snippets.

After validator setup, ``window[0]`` is the transcript cut to 1000 characters
and later lines are ``title\\nsnippet``. A raw JSON bundle, used in local
replay, is unpacked the same way ``NamedEntitiesExtractionTaskBundle`` does.
"""

from __future__ import annotations

import json
from typing import Any

_TRANSCRIPT_CHARS = 1000
_SNIPPET_CHARS = 1000
_MAX_SNIPPETS = 6


def _line_text(line: Any) -> str:
    if isinstance(line, (list, tuple)) and len(line) >= 2:
        return str(line[1] or "")
    return str(line or "")


def _snippet_from_hit(hit: dict) -> str:
    title = str(hit.get("title") or "").strip()
    snippet = str(hit.get("snippet") or "").strip()
    text = f"{title}\n{snippet}".strip()
    return text[:_SNIPPET_CHARS]


def unpack_ner_json(text: str) -> tuple[str, list[str]]:
    """Return ``(transcript, snippets)`` when ``text`` is a raw NER JSON blob."""
    raw = (text or "").strip()
    if not raw.startswith("{"):
        return "", []
    try:
        parsed = json.loads(raw)
    except Exception:
        return "", []
    if not isinstance(parsed, dict) or "transcript_text" not in parsed:
        return "", []
    transcript = str(parsed.get("transcript_text") or "")[:_TRANSCRIPT_CHARS]
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
                if len(snippets) >= _MAX_SNIPPETS:
                    return transcript, snippets
    return transcript, snippets


def parse_ner_task(task) -> tuple[str, list[str]]:
    """Return ``(transcript, snippets)`` from a mining task window."""
    data = getattr(getattr(task, "input", None), "data", None)
    window = list(getattr(data, "window", None) or [])
    texts = [_line_text(line).strip() for line in window]
    texts = [text for text in texts if text]
    if not texts:
        return "", []

    transcript, snippets = unpack_ner_json(texts[0])
    if transcript or snippets:
        if len(texts) > 1:
            # The validator already chose which snippets to send. Prefer those.
            chosen = [text[:_SNIPPET_CHARS] for text in texts[1:] if text.strip()]
            if chosen:
                snippets = chosen
        return transcript[:_TRANSCRIPT_CHARS], snippets[:_MAX_SNIPPETS]

    transcript = texts[0][:_TRANSCRIPT_CHARS]
    snippets = [text[:_SNIPPET_CHARS] for text in texts[1:] if text.strip()]
    return transcript, snippets[:_MAX_SNIPPETS]
