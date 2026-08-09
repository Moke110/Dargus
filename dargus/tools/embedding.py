"""Embedding tool — embed/test/info operations over the session ToolCache.

The heavy embedding model is loaded once into the session
:class:`~dargus.tools.cache.ToolCache` and reused across PRA rounds
(design/3_D-Base.md §embedding tool).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.models.embedding import EmbeddingModel
    from dargus.tools.cache import ToolCache

_CACHE_KEY = "embedding_model"


def _get_model(cache: ToolCache | None) -> EmbeddingModel:
    """Return the cached EmbeddingModel, creating the default on first use."""
    if cache is not None:
        return cache.get(_CACHE_KEY, _default_model)
    return _default_model()


def _default_model() -> EmbeddingModel:
    from dargus.models.embedding import EmbeddingModel, SentenceTransformerBackend

    return EmbeddingModel(SentenceTransformerBackend("all-MiniLM-L6-v2"))


def embedding(
    texts: list[str] | None = None,
    op: str = "embed",
    cache: ToolCache | None = None,
) -> dict[str, Any]:
    """Embed texts or inspect the embedding model.

    Args:
        texts: Texts to embed (``op="embed"`` and ``op="test"``).
        op: ``"embed"`` (vectors for *texts*), ``"test"`` (embed a probe
            text and report success), or ``"info"`` (model metadata).
        cache: Session ToolCache holding the resident model.

    Returns:
        Dict with ``op`` and, per op: ``vectors`` (embed), ``ok`` (test),
        or ``model_name`` / ``dimension`` (info).
    """
    model = _get_model(cache)

    if op == "info":
        return {"op": "info", "model_name": model.model_name}

    if op == "test":
        probe = texts or ["dargus embedding self-test"]
        vectors = model.embed(probe)
        return {
            "op": "test",
            "ok": True,
            "model_name": model.model_name,
            "dimension": len(vectors[0]) if vectors else 0,
        }

    if op == "embed":
        if not texts:
            return {"op": "embed", "vectors": [], "model_name": model.model_name}
        return {
            "op": "embed",
            "vectors": model.embed(list(texts)),
            "model_name": model.model_name,
        }

    raise ValueError(f"Unknown embedding op {op!r}. Valid: embed, test, info")
