"""Segment features. The original six (is_system, is_last_turn, token_count_norm, recency_norm,
repeat_count, query_similarity) are the core and can be ablated to on their own."""
from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from sieve.segmenter import KINDS, Segment

CORE_FEATURES = ["is_system", "is_last_turn", "token_count_norm", "recency_norm", "repeat_count", "query_similarity"]
EXTRA_FEATURES = [f"kind_{k}" for k in KINDS] + ["digit_density", "near_dup_score", "prior_expand_count"]
ALL_FEATURES = CORE_FEATURES + EXTRA_FEATURES


class Similarity:
    """Cosine(segment, query): MiniLM on CPU, TF-IDF fallback (flagged)."""

    def __init__(self, model_name: str, force_tfidf: bool = False):
        self.method = "tfidf"
        self._model = None
        self._memo: "OrderedDict[bytes, np.ndarray]" = OrderedDict()  # hashed keys, bounded LRU
        self.load_error: Optional[str] = None
        if not force_tfidf:
            try:
                import logging
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, device="cpu")
                self.method = "minilm"
            except Exception as e:  # pragma: no cover
                self.load_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _k(t: str) -> bytes:
        return hashlib.blake2b(t.encode("utf-8"), digest_size=16).digest()

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        keys = [self._k(t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._memo]
        if todo:
            vecs = self._model.encode([t for _, t in todo], batch_size=32, normalize_embeddings=True,
                                      show_progress_bar=False)
            for (k, _), v in zip(todo, vecs):
                self._memo[k] = np.asarray(v, dtype=np.float32)
            while len(self._memo) > 50_000:
                self._memo.popitem(last=False)
        out = []
        for k in keys:
            self._memo.move_to_end(k)
            out.append(self._memo[k])
        return out

    def scores(self, texts: List[str], query: str) -> List[float]:
        if not texts:
            return []
        if self._model is not None:
            embs = self._embed(texts + [query])
            q = embs[-1]
            return [float(max(0.0, np.dot(e, q))) for e in embs[:-1]]
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-.]*\b", sublinear_tf=True)
        try:
            m = vec.fit_transform(texts + [query])
        except ValueError:
            return [0.0] * len(texts)
        q = m[len(texts)]
        return [float((m[i].multiply(q)).sum()) for i in range(len(texts))]


def compute_features(segs: List[Segment], query: str, current_turn: int, sim: Similarity,
                     expand_counts: Optional[Dict[str, int]] = None) -> None:
    expand_counts = expand_counts or {}
    sims = sim.scores([s.content for s in segs], query)
    last_user_turn = max([s.turn_index for s in segs if s.kind != "query"] or [0])
    for s, q in zip(segs, sims):
        digits = sum(ch.isdigit() for ch in s.content)
        f = {
            "is_system": float(s.role == "system"),
            "is_last_turn": float(s.turn_index >= last_user_turn),
            "token_count_norm": min(1.0, s.tokens / 400.0),
            "recency_norm": min(1.0, (current_turn - s.turn_index) / 10.0),  # turns ago / 10
            "repeat_count": float(s.repeat_count),
            "query_similarity": round(q, 4),
            "digit_density": round(digits / max(1, len(s.content)), 4),
            "near_dup_score": round(s.near_dup_score, 4),
            "prior_expand_count": float(expand_counts.get(s.id, 0)),
            "turns_old": float(current_turn - s.turn_index),
        }
        for k in KINDS:
            f[f"kind_{k}"] = float(s.kind == k)
        s.features = f


def vectorize(s: Segment, names: List[str] = ALL_FEATURES) -> List[float]:
    return [float(s.features.get(n, 0.0)) for n in names]
