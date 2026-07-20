from __future__ import annotations

import numpy as np


class MockNLP:
    """Fallback embedder returning zero vectors when sentence-transformers is unavailable."""

    def embed_text(self, text: str) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        return 0.0


class DBaseNLP:
    """Embedding backend for D-Base records and DiseaseRAG knowledge bases."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:  # noqa: BLE001
                return MockNLP().embed_text(text)
        try:
            return self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        except Exception:  # noqa: BLE001
            return MockNLP().embed_text(text)

    def similarity(self, a: str, b: str) -> float:
        va = self.embed_text(a)
        vb = self.embed_text(b)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(np.dot(va, vb) / norm)
