"""Dialogue topic segmentation via turn cohesion (BERTSeg-lite).

Splits conversation into 2–6 coherent segments without historical GT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPEAKER = re.compile(r"^(?:speaker\s*\d+|user|assistant|host|guest)\s*[:\-]\s*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Segment:
    idx: int
    text: str
    turn_start: int
    turn_end: int


def _turns(text: str) -> list[str]:
    """Split into utterance-sized turns."""
    raw = (text or "").strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 3:
        turns = [_SPEAKER.sub("", ln).strip() for ln in lines]
        return [t for t in turns if t]
    # Single blob: sentence pack into pseudo-turns of ~2 sentences
    sents = [s.strip() for s in _SENT_SPLIT.split(raw) if s and s.strip()]
    if not sents:
        return [raw]
    turns: list[str] = []
    buf: list[str] = []
    for s in sents:
        buf.append(s)
        if len(buf) >= 2:
            turns.append(" ".join(buf))
            buf = []
    if buf:
        turns.append(" ".join(buf))
    return turns


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def segment_dialogue(
    primary: str,
    enrichment: str = "",
    *,
    min_segments: int = 2,
    max_segments: int = 6,
    break_thr: float = 0.12,
) -> list[Segment]:
    """Segment primary dialogue; attach enrichment as its own trailing segment."""
    turns = _turns(primary)
    if not turns:
        body = (enrichment or "").strip()
        return [Segment(0, body, 0, 0)] if body else []

    # Cohesion depth between adjacent turns
    sets = [_token_set(t) for t in turns]
    depths: list[float] = []
    for i in range(len(sets) - 1):
        sim = _jaccard(sets[i], sets[i + 1])
        # Also peek at window of ±1 for smoother TextTiling-style depth
        left = _jaccard(sets[max(0, i - 1)], sets[i])
        right = _jaccard(sets[i + 1], sets[min(len(sets) - 1, i + 2)])
        depth = max(0.0, (left + right) / 2.0 - sim)
        depths.append(depth)

    # Pick break points at local maxima above threshold, capped
    candidates: list[tuple[float, int]] = []
    for i, d in enumerate(depths):
        if d < break_thr:
            continue
        left_ok = i == 0 or d >= depths[i - 1]
        right_ok = i == len(depths) - 1 or d >= depths[i + 1]
        if left_ok and right_ok:
            candidates.append((d, i + 1))  # break before turn i+1
    candidates.sort(reverse=True)
    max_breaks = max(0, max_segments - 1 - (1 if enrichment.strip() else 0))
    breaks = sorted({idx for _, idx in candidates[:max_breaks]})

    # Ensure minimum segments by evenly splitting if needed
    if len(breaks) + 1 < min_segments and len(turns) >= min_segments:
        step = max(1, len(turns) // min_segments)
        breaks = sorted({min(len(turns) - 1, step * i) for i in range(1, min_segments)})
        breaks = [b for b in breaks if 0 < b < len(turns)]

    segments: list[Segment] = []
    start = 0
    for b in breaks + [len(turns)]:
        chunk = " ".join(turns[start:b]).strip()
        if chunk:
            segments.append(Segment(len(segments), chunk, start, b - 1))
        start = b

    enr = (enrichment or "").strip()
    if enr:
        # Cap enrichment segment size for linking budget
        segments.append(Segment(len(segments), enr[:4000], -1, -1))

    if not segments:
        combined = (primary + "\n" + enrichment).strip()
        return [Segment(0, combined[:6000], 0, 0)] if combined else []

    # Merge tiny leading/trailing chatter segments into neighbors
    merged: list[Segment] = []
    for seg in segments:
        if merged and len(seg.text) < 80 and seg.turn_start >= 0:
            prev = merged[-1]
            merged[-1] = Segment(prev.idx, (prev.text + " " + seg.text).strip(), prev.turn_start, seg.turn_end)
        else:
            merged.append(Segment(len(merged), seg.text, seg.turn_start, seg.turn_end))
    return merged[:max_segments]
