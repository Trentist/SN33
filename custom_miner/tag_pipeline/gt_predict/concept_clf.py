"""Concept classifier: window → frequent GT concept probabilities.

Default: TF-IDF + OneVsRest logistic (fast, fits small labeled sets).
Optional: ModernBERT mean-pool encoder when HYBRID_USE_MODERNBERT=1 and
transformers can load the model (slower; use offline warmup).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "data"
_CLF_PATH = _DATA / "concept_clf.joblib"
_LABELS_PATH = _DATA / "concept_clf_labels.json"

_clf: "ConceptClassifier | None" = None


class ConceptClassifier:
    def __init__(self) -> None:
        self._pipe = None
        self._labels: list[str] = []
        self._ready = False
        self._modernbert = None
        self._use_bert = os.environ.get("HYBRID_USE_MODERNBERT", "0") == "1"
        self._load()

    def _load(self) -> None:
        try:
            import joblib
        except ImportError:
            return
        if not (_CLF_PATH.exists() and _LABELS_PATH.exists()):
            return
        try:
            self._pipe = joblib.load(_CLF_PATH)
            self._labels = json.loads(_LABELS_PATH.read_text())
            self._ready = bool(self._pipe is not None and self._labels)
            if self._use_bert:
                self._maybe_load_modernbert()
        except Exception:
            self._ready = False

    def _maybe_load_modernbert(self) -> None:
        """Lazy-load ModernBERT encoder if requested (optional path)."""
        try:
            from transformers import AutoModel
            from transformers import AutoTokenizer
            import torch

            name = os.environ.get("HYBRID_MODERNBERT_MODEL", "answerdotai/ModernBERT-base")
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            model.eval()
            self._modernbert = (tok, model, torch)
        except Exception:
            self._modernbert = None

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, window_text: str, *, top_k: int = 12, min_p: float = 0.12) -> list[tuple[str, float]]:
        if not self._ready or not (window_text or "").strip():
            return []
        assert self._pipe is not None
        try:
            if hasattr(self._pipe, "predict_proba"):
                proba = self._pipe.predict_proba([window_text])
            else:
                return []
        except Exception:
            return []

        # OneVsRest returns list of arrays; Pipeline with MultiOutput differs
        scores: list[float] = []
        if isinstance(proba, list):
            # list of (n_samples, 2) per label
            for arr in proba:
                a = np.asarray(arr)
                if a.ndim == 2 and a.shape[1] >= 2:
                    scores.append(float(a[0, -1]))
                else:
                    scores.append(float(a.ravel()[0]))
        else:
            a = np.asarray(proba)
            if a.ndim == 2:
                scores = [float(x) for x in a[0]]
            else:
                return []

        ranked = sorted(
            ((self._labels[i], scores[i]) for i in range(min(len(self._labels), len(scores)))),
            key=lambda kv: -kv[1],
        )
        return [(t, p) for t, p in ranked if p >= min_p][:top_k]


def get_concept_classifier() -> ConceptClassifier:
    global _clf
    if _clf is None:
        _clf = ConceptClassifier()
    return _clf


def rebuild_classifier_from_rows(rows: list[dict[str, Any]], *, min_tag_count: int = 2) -> dict[str, Any]:
    """Train multi-label concept classifier from [{window, tags}, ...]."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MultiLabelBinarizer

    texts = [r["window"] for r in rows if r.get("window") and r.get("tags")]
    tag_lists = [list(r["tags"]) for r in rows if r.get("window") and r.get("tags")]
    if len(texts) < 4:
        return {"ok": False, "n": len(texts), "error": "need >=4 windows"}

    # Keep labels that appear often enough
    from collections import Counter

    cnt = Counter(t for ts in tag_lists for t in ts)
    keep = {t for t, c in cnt.items() if c >= min_tag_count}
    filtered = [[t for t in ts if t in keep] for ts in tag_lists]
    # Drop empty
    pairs = [(x, y) for x, y in zip(texts, filtered) if y]
    if len(pairs) < 4:
        return {"ok": False, "n": len(pairs), "error": "too few after label filter"}
    texts = [p[0] for p in pairs]
    tag_lists = [p[1] for p in pairs]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tag_lists)
    labels = list(mlb.classes_)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8192,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear"),
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(texts, Y)

    _DATA.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, _CLF_PATH)
    _LABELS_PATH.write_text(json.dumps(labels, indent=2))
    global _clf
    _clf = None
    return {"ok": True, "n": len(texts), "n_labels": len(labels), "path": str(_CLF_PATH)}
