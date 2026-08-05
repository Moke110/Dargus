"""Shared test fixtures for all Dargus tests."""

import pytest


@pytest.fixture
def minimal_dbase(tmp_path):
    """Set up a D-Base with minimal data for testing."""
    import os

    dargus_home = str(tmp_path / "dargus_home")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)
    return dargus_home


@pytest.fixture(autouse=True)
def _no_real_api_key():
    """Prevent a real DARGUS_LLM_API_KEY from leaking into the suite.

    Model-driven paths (predict, T8) call the reasoning LLM; a leaked key
    would trigger real, slow network calls in tests. Tests drive the stub
    path instead. Tests that need a key can set it explicitly.
    """
    import os

    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


@pytest.fixture(autouse=True)
def _no_real_embedding_model(request, monkeypatch):
    """Block the lazy SentenceTransformer default from loading in tests.

    The real backend performs a Hugging Face hub call on first embed, which
    hangs when the model is not reachable locally. Tests that exercise the
    real backend can opt out with ``@pytest.mark.real_embedding``.
    """
    if request.node.get_closest_marker("real_embedding"):
        return
    import numpy as np

    from dargus.models.embedding import EmbeddingModel

    class _HashEmbeddingBackend:
        """Deterministic offline stand-in for the default embedding backend."""

        def embed(self, texts):
            vectors = []
            for text in texts:
                vec = np.zeros(384, dtype=np.float32)
                for i, b in enumerate(text.encode("utf-8")):
                    vec[i % 384] += float(b)
                norm = np.linalg.norm(vec)
                vectors.append((vec / norm if norm else vec).tolist())
            return vectors

    def _fake_lazy_model(_store_self):
        if _store_self._embedding_model is None:
            _store_self._embedding_model = EmbeddingModel(_HashEmbeddingBackend())
        return _store_self._embedding_model

    monkeypatch.setattr("dargus.dbase.store.DBaseStore._get_embedding_model", _fake_lazy_model)
    monkeypatch.setattr(
        "dargus.models.embedding.SentenceTransformerBackend._ensure_model",
        lambda _backend_self: pytest.fail(
            "SentenceTransformerBackend tried to load the real model during "
            "tests; inject a stub backend or mark the test real_embedding"
        ),
    )
