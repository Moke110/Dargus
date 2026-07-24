"""EmbeddingModel — text embedding and similarity computation."""

from __future__ import annotations

import logging
import math
from typing import Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Embedding = list[float]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends — the actual embedding model invocation."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Return a list of embedding vectors, one per input text."""
        ...


# ---------------------------------------------------------------------------
# SentenceTransformer backend
# ---------------------------------------------------------------------------


class SentenceTransformerBackend:
    """Default embedding backend using sentence-transformers (all-MiniLM-L6-v2).

    The model is lazy-loaded on the first call to ``embed()``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading SentenceTransformer model '%s' ...", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{self._model_name}': {exc}"
            ) from exc

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts and return their vector representations."""
        self._ensure_model()
        from sentence_transformers import SentenceTransformer

        assert self._model is not None
        model: SentenceTransformer = self._model  # type: ignore[assignment]
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# EmbeddingModel facade
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Unified embedding interface used across the runtime.

    Delegates to an EmbeddingBackend and provides similarity computation.
    """

    def __init__(self, backend: EmbeddingBackend) -> None:
        self._backend = backend

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts."""
        return self._backend.embed(texts)

    @staticmethod
    def similarity(a: Embedding, b: Embedding) -> float:
        """Compute cosine similarity between two embedding vectors.

        Uses pure Python math — no numpy dependency for this single computation.
        Returns 0.0 if either vector is all zeros.
        """
        if len(a) != len(b):
            raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)
