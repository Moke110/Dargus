"""Tests for EmbeddingModel, EmbeddingBackend, and cosine similarity."""

import math

import pytest

from dargus.models.embedding import (
    Embedding,
    EmbeddingModel,
    SentenceTransformerBackend,
)


class MockEmbeddingBackend:
    """Mock EmbeddingBackend that returns fixed-size vectors."""

    def __init__(self, dimension: int = 8):
        self._dim = dimension
        self.call_count = 0
        self.last_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[Embedding]:
        self.call_count += 1
        self.last_texts = texts
        # Return deterministic vectors based on text length for testability
        result = []
        for i, t in enumerate(texts):
            v = [0.0] * self._dim
            v[i % self._dim] = float(len(t))
            result.append(v)
        return result


class TestSentenceTransformerBackend:
    """Tests for SentenceTransformerBackend constructor."""

    def test_default_model_name(self):
        backend = SentenceTransformerBackend()
        assert backend._model_name == "all-MiniLM-L6-v2"
        assert backend._model is None  # not loaded yet

    def test_custom_model_name(self):
        backend = SentenceTransformerBackend(model_name="custom-model")
        assert backend._model_name == "custom-model"


class TestEmbeddingModel:
    """Tests for EmbeddingModel facade."""

    def test_embed_delegates_to_backend(self):
        backend = MockEmbeddingBackend(dimension=4)
        model = EmbeddingModel(backend)
        texts = ["hello", "world"]
        result = model.embed(texts)

        assert backend.call_count == 1
        assert backend.last_texts == texts
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_embed_single_text(self):
        backend = MockEmbeddingBackend(dimension=4)
        model = EmbeddingModel(backend)
        result = model.embed(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 4


class TestCosineSimilarity:
    """Tests for EmbeddingModel.similarity()."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        sim = EmbeddingModel.similarity(v, v)
        assert math.isclose(sim, 1.0, rel_tol=1e-9)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = EmbeddingModel.similarity(a, b)
        assert math.isclose(sim, 0.0, abs_tol=1e-9)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = EmbeddingModel.similarity(a, b)
        assert math.isclose(sim, -1.0, rel_tol=1e-9)

    def test_known_similarity(self):
        # cos(theta) = (1*2 + 2*3 + 3*4) / (sqrt(1+4+9) * sqrt(4+9+16))
        # = (2+6+12) / (sqrt(14) * sqrt(29)) = 20 / sqrt(406) ≈ 0.992278
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        expected = 20.0 / math.sqrt(14.0 * 29.0)
        sim = EmbeddingModel.similarity(a, b)
        assert math.isclose(sim, expected, rel_tol=1e-6)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        sim = EmbeddingModel.similarity(a, b)
        assert sim == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            EmbeddingModel.similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_float_similarity(self):
        v = [float(i + 1) for i in range(128)]
        sim = EmbeddingModel.similarity(v, v)
        assert math.isclose(sim, 1.0, rel_tol=1e-9)
