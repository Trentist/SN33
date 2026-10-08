#!/usr/bin/env python3
"""Build FAISS window index + concept classifier from prepared_bundles.

Simulates the validator mask: each indexed window (or synthetic slice) maps to
full-conversation GT tags.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PREP = _ROOT / "local_dev" / "cache" / "miner_problems" / "prepared_bundles"


def _line_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[1])
    if isinstance(item, dict):
        return str(item.get("text") or item.get("content") or "")
    return str(item)


def collect_rows(prep: Path = PREP) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(prep.glob("*conversation_tagging*.json")):
        b = json.loads(path.read_text())
        inp = b.get("input") or {}
        meta = inp.get("metadata") or {}
        tags = [str(t).strip().lower() for t in (meta.get("tags") or []) if t]
        if not tags:
            continue
        data = inp.get("data") or {}
        lines = data.get("lines") or []
        texts = [_line_text(x).strip() for x in lines]
        texts = [t for t in texts if t]

        # Indexed windows (validator-style slices)
        for win in data.get("indexed_windows") or []:
            if isinstance(win, dict):
                wlines = win.get("lines") or win.get("window") or []
            else:
                wlines = win
            parts = [_line_text(x).strip() for x in (wlines or [])]
            parts = [p for p in parts if p]
            if not parts:
                continue
            rows.append({"window": "\n".join(parts), "tags": tags, "source": path.name})

        # Synthetic random 2–10 turn windows from full conversation
        if len(texts) >= 2:
            for _ in range(min(8, max(2, len(texts) // 2))):
                n = random.randint(2, min(10, len(texts)))
                start = random.randint(0, max(0, len(texts) - n))
                chunk = texts[start : start + n]
                rows.append({"window": "\n".join(chunk), "tags": tags, "source": path.name + ":synth"})

        # Enrichment-augmented windows (enrichment is strong GT signal)
        enr = data.get("enrichment_lines") or data.get("enrichment") or []
        enr_bits: list[str] = []
        for item in enr:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                enr_bits.append(str(item[1]))
            elif isinstance(item, str):
                enr_bits.append(item)
            elif isinstance(item, dict):
                enr_bits.append(str(item.get("content") or item.get("text") or ""))
        enr_text = "\n".join(b for b in enr_bits if b).strip()
        if texts and enr_text:
            mid = texts[len(texts) // 4 : len(texts) // 4 + 4] or texts[:4]
            rows.append(
                {
                    "window": "\n".join(mid) + "\n" + enr_text[:1200],
                    "tags": tags,
                    "source": path.name + ":enr",
                }
            )

    # Dedupe identical windows
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        key = r["window"][:500]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def main() -> None:
    random.seed(33)
    rows = collect_rows()
    print(f"collected {len(rows)} window→GT rows")
    from custom_miner.tag_pipeline.gt_predict.faiss_retriever import rebuild_faiss_from_rows
    from custom_miner.tag_pipeline.gt_predict.concept_clf import rebuild_classifier_from_rows

    fa = rebuild_faiss_from_rows(rows)
    print("faiss:", fa)
    clf = rebuild_classifier_from_rows(rows, min_tag_count=2)
    print("classifier:", clf)


if __name__ == "__main__":
    main()
