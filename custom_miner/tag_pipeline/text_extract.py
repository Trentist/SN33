"""Pull miner-visible text from a Task (window + enrichment)."""

from __future__ import annotations

import json
import re
from typing import Any

from custom_miner.tag_pipeline.anchors import clean_text

_STUB_LINE = re.compile(
    r"(this domain is for use in documentation|example\.com|example\.org|"
    r"avoid use in operations|lorem ipsum)",
    re.I,
)


def _line_text(line: Any) -> str:
    if isinstance(line, (list, tuple)) and len(line) >= 2:
        return str(line[1] or "")
    return str(line or "")


def _is_stub(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # Short chrome-only stubs; keep short but substantive technical terms
    if len(t) < 40 and _STUB_LINE.search(t):
        return True
    if _STUB_LINE.search(t) and len(t) < 200:
        return True
    return False


def _unpack_webpage_json(text: str) -> tuple[str, str]:
    """If window line is raw webpage JSON, split markdown vs search-result enrichment."""
    t = (text or "").strip()
    if not t.startswith("{"):
        return "", ""
    try:
        parsed = json.loads(t)
    except Exception:
        return "", ""
    if not isinstance(parsed, dict):
        return "", ""
    md = str(parsed.get("website_markdown") or "")[:2000]
    enrich_parts: list[str] = []
    enr = parsed.get("enrichment") or {}
    results = enr.get("search_results") if isinstance(enr, dict) else None
    if isinstance(results, dict):
        for query, hits in results.items():
            if isinstance(query, str) and query.strip():
                enrich_parts.append(query.strip())
            if not isinstance(hits, list):
                continue
            for hit in hits[:5]:
                if not isinstance(hit, dict):
                    continue
                title = str(hit.get("title") or "").strip()
                snippet = str(hit.get("snippet") or "").strip()
                if title or snippet:
                    enrich_parts.append(f"{title}\n{snippet}"[:1000])
    return md, "\n".join(enrich_parts)


def extract_task_texts(task) -> dict[str, str]:
    """Return primary / enrichment / combined / selection texts.

    Conversation: primary=window, enrichment=enrichment_lines.
    NER: primary=longest non-stub window line, rest=enrichment.
    Webpage: skip example.com stubs; use enrichment snippets as primary signal.
    """
    data = getattr(getattr(task, "input", None), "data", None)
    empty = {"primary": "", "enrichment": "", "combined": "", "selection_text": ""}
    if data is None:
        return empty

    window = getattr(data, "window", None) or []
    enrichment_lines = getattr(data, "enrichment_lines", None) or []
    task_type = getattr(task, "type", "")

    win_texts = [_line_text(x) for x in window]
    win_texts = [t for t in win_texts if t and t.strip()]

    if task_type == "webpage_metadata_generation":
        primary_parts: list[str] = []
        enrich_parts: list[str] = []
        for t in win_texts:
            md, nested = _unpack_webpage_json(t)
            if md or nested:
                if md and not _is_stub(md):
                    primary_parts.append(md)
                if nested:
                    enrich_parts.append(nested)
                continue
            if _is_stub(t):
                continue
            # Non-stub window lines on webpage tasks are usually enrichment snippets
            enrich_parts.append(t)
        # Prefer enrichment (search results) as the GT-aligned primary body
        body = enrich_parts or primary_parts or [t for t in win_texts if not _is_stub(t)] or win_texts
        primary = clean_text("\n".join(body))
        enrichment = clean_text("\n".join(enrich_parts if primary_parts else []))
        if not enrichment and enrich_parts and primary_parts:
            enrichment = clean_text("\n".join(enrich_parts))
        # Selection: enrichment-heavy technical corpus
        selection_text = (primary[:3500] + "\n" + enrichment[:1000]).strip()
        combined = (primary + "\n" + enrichment).strip()
        return {
            "primary": primary,
            "enrichment": enrichment or primary,
            "combined": combined or primary,
            "selection_text": selection_text or combined or primary,
        }

    if task_type == "named_entities_extraction" and win_texts:
        ranked = sorted(win_texts, key=lambda t: (0 if _is_stub(t) else 1, len(t)), reverse=True)
        primary_parts = [ranked[0]] if ranked else []
        enrich_parts = [t for t in win_texts if t != primary_parts[0]] + [
            _line_text(x) for x in enrichment_lines
        ]
    else:
        primary_parts = win_texts
        enrich_parts = [_line_text(x) for x in enrichment_lines]

    primary = clean_text("\n".join(p for p in primary_parts if p.strip()))
    enrichment = clean_text("\n".join(p for p in enrich_parts if p.strip()))
    # Selection text for embeddings: enrichment first when it has real entities
    if enrichment and len(enrichment) > 200:
        selection_text = (enrichment[:2000] + "\n" + primary[:1500]).strip()
    else:
        selection_text = (primary + "\n" + enrichment[:800]).strip()
    combined = (primary + "\n" + enrichment).strip()
    return {
        "primary": primary,
        "enrichment": enrichment,
        "combined": combined,
        "selection_text": selection_text or combined,
    }
