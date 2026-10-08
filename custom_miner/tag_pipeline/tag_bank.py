"""Lexical tag-bank retrieval (no network). Seeded from historical GT tags."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

_BANK_PATH = Path(__file__).resolve().parent / "data" / "tag_bank.json"
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']{1,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


@dataclass
class TagBank:
    tags: list[str] = field(default_factory=list)
    _token_index: dict[str, set[int]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | None = None) -> "TagBank":
        path = path or _BANK_PATH
        tags: list[str] = []
        if path.is_file():
            try:
                raw = json.loads(path.read_text())
                tags = [str(t).strip().lower() for t in (raw.get("tags") or []) if str(t).strip()]
            except Exception:
                tags = []
        bank = cls(tags=tags)
        bank._build_index()
        return bank

    def _build_index(self) -> None:
        self._token_index = {}
        for i, tag in enumerate(self.tags):
            for tok in _tokens(tag):
                self._token_index.setdefault(tok, set()).add(i)

    def retrieve(self, query_text: str, *, k: int = 16) -> list[str]:
        """Token-overlap kNN against the bank."""
        q = _tokens(query_text)
        if not q or not self.tags:
            return []

        scores: dict[int, float] = {}
        for tok in q:
            for idx in self._token_index.get(tok, ()):
                tag_toks = _tokens(self.tags[idx])
                if not tag_toks:
                    continue
                # Jaccard-ish incremental
                scores[idx] = scores.get(idx, 0.0) + 1.0 / max(len(tag_toks), 1)

        # Finalize with true Jaccard for top candidates
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: k * 3]
        final: list[tuple[float, str]] = []
        for idx, _ in ranked:
            tag = self.tags[idx]
            tset = _tokens(tag)
            inter = len(q & tset)
            union = len(q | tset) or 1
            # Prefer shorter entity-like tags with solid overlap
            j = inter / union
            final.append((j + 0.05 * inter, tag))

        final.sort(key=lambda x: -x[0])
        out: list[str] = []
        seen: set[str] = set()
        for score, tag in final:
            if tag in seen:
                continue
            tset = _tokens(tag)
            overlap = len(q & tset)
            # Short brands (ufc, marvel, claude): exact token presence is enough
            if len(tset) == 1:
                if overlap < 1:
                    continue
            elif len(tset) >= 4:
                # Long GT phrases need strong overlap (block cross-domain leaks)
                if overlap < 3 and score < 0.45:
                    continue
            elif overlap < 2 and score < 0.35:
                # Require real overlap — avoid weak 1-token leaks across domains
                continue
            if len(tag.split()) > 8:
                continue
            seen.add(tag)
            out.append(tag)
            if len(out) >= k:
                break
        return out


_CACHED: TagBank | None = None


def get_tag_bank() -> TagBank:
    global _CACHED
    if _CACHED is None:
        _CACHED = TagBank.load()
    return _CACHED
