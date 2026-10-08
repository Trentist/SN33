"""FAISS retrieval: window embedding → similar historical windows → their GT tags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "data"
_INDEX_PATH = _DATA / "faiss_windows.index"
_META_PATH = _DATA / "faiss_windows_meta.json"
_VECTORIZER_PATH = _DATA / "faiss_vectorizer.joblib"

_retriever: "FaissGtRetriever | None" = None


class FaissGtRetriever:
    """Retrieve GT tags from nearest training windows."""

    def __init__(self) -> None:
        self._index = None
        self._vectorizer = None
        self._gt_lists: list[list[str]] = []
        self._ready = False
        self._load()

    def _load(self) -> None:
        try:
            import faiss
            import joblib
        except ImportError:
            return
        if not (_INDEX_PATH.exists() and _META_PATH.exists() and _VECTORIZER_PATH.exists()):
            return
        try:
            self._index = faiss.read_index(str(_INDEX_PATH))
            self._vectorizer = joblib.load(_VECTORIZER_PATH)
            meta = json.loads(_META_PATH.read_text())
            self._gt_lists = meta.get("gt_lists") or []
            self._ready = bool(self._index and self._gt_lists)
        except Exception:
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def retrieve(self, window_text: str, *, k: int = 20, tag_limit: int = 20) -> list[tuple[str, float]]:
        """Return (tag, support_score) from top-k neighbor windows' GT tags."""
        if not self._ready or not (window_text or "").strip():
            return []
        assert self._vectorizer is not None and self._index is not None
        try:
            import faiss
        except ImportError:
            return []

        x = self._vectorizer.transform([window_text]).astype(np.float32)
        if hasattr(x, "toarray"):
            x = x.toarray()
        faiss.normalize_L2(x)
        k = min(k, int(self._index.ntotal))
        if k <= 0:
            return []
        sims, idxs = self._index.search(x, k)
        votes: dict[str, float] = {}
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(self._gt_lists):
                continue
            w = float(max(sim, 0.0))
            for tag in self._gt_lists[idx]:
                votes[tag] = votes.get(tag, 0.0) + w
        if not votes:
            return []
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        # Normalize support to [0, 1]
        top = ranked[0][1] or 1.0
        return [(t, min(s / top, 1.0)) for t, s in ranked[:tag_limit]]


def get_faiss_retriever() -> FaissGtRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FaissGtRetriever()
    return _retriever


def rebuild_faiss_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build FAISS index from [{window, tags}, ...]. Returns stats."""
    import faiss
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [r["window"] for r in rows if r.get("window") and r.get("tags")]
    gt_lists = [list(r["tags"]) for r in rows if r.get("window") and r.get("tags")]
    if len(texts) < 2:
        return {"ok": False, "n": len(texts), "error": "need >=2 windows"}

    vectorizer = TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
    )
    mat = vectorizer.fit_transform(texts).astype(np.float32)
    dense = mat.toarray()
    faiss.normalize_L2(dense)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    _DATA.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_PATH))
    joblib.dump(vectorizer, _VECTORIZER_PATH)
    _META_PATH.write_text(
        json.dumps({"n": len(texts), "dim": int(dense.shape[1]), "gt_lists": gt_lists}, indent=2)
    )
    global _retriever
    _retriever = None
    return {"ok": True, "n": len(texts), "dim": int(dense.shape[1]), "path": str(_INDEX_PATH)}
