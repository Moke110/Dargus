"""Tests for D-Base Tool wrappers."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from dargus.dbase import DBase
from dargus.dbase.store import DBaseStore
from dargus.models.embedding import Embedding, EmbeddingBackend, EmbeddingModel
from dargus.tools.dbase import dbase_query, dbase_status, dbase_write

# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------


class _StubEmbeddingBackend(EmbeddingBackend):
    """Stub backend that returns distinct embeddings per text for testing."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        embeddings: list[Embedding] = []
        for text in texts:
            # Quadratic formula on a hash seed so each dimension varies within
            # the vector and vectors differ across texts.  This avoids constant
            # vectors that all have cosine similarity 1.0 with each other.
            seed = abs(hash(text)) % 10000
            emb: list[float] = []
            for i in range(384):
                emb.append(0.01 * ((seed + i) % 100))
            embeddings.append(emb)
        return embeddings


def _make_evidence(**overrides):
    """Return a valid v1.0.0 three-axis evidence dict (descriptive, xy.count=1)."""
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {
            "type": "logP",
            "category": "pk_adme",
            "value": [3.5],
            "assay": "binding_assay",
        },
        "bg": {"disease_id": [], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


def _new_manager_with_stub():
    """Create a DBaseStore backed by a tempdir DBase with a stub embedding model."""
    tmp = tempfile.TemporaryDirectory()
    dbase = DBase("test", root_dir=tmp.name)
    emb_model = EmbeddingModel(_StubEmbeddingBackend())
    manager = DBaseStore(dbase, embedding_model=emb_model)
    return manager, tmp


def _new_manager_without_embedding():
    """Create a DBaseStore backed by a tempdir DBase, no embedding model."""
    tmp = tempfile.TemporaryDirectory()
    dbase = DBase("test", root_dir=tmp.name)
    manager = DBaseStore(dbase)
    return manager, tmp


# ---------------------------------------------------------------------------
# dbase_query
# ---------------------------------------------------------------------------


def test_dbase_query_returns_correct_format():
    manager, tmp = _new_manager_with_stub()
    try:
        manager.write_record(_make_evidence())
        result = dbase_query(manager, {})
        assert isinstance(result, dict)
        assert "records" in result
        assert "count" in result
        assert isinstance(result["records"], list)
        assert isinstance(result["count"], int)
        assert result["count"] == len(result["records"])
    finally:
        tmp.cleanup()


def test_dbase_query_respects_limit():
    manager, tmp = _new_manager_with_stub()
    try:
        manager.write_record(
            _make_evidence(
                y={"type": "a", "category": "pk_adme", "value": [1.0], "assay": "binding_assay"}
            )
        )
        manager.write_record(
            _make_evidence(
                y={"type": "b", "category": "pk_adme", "value": [2.0], "assay": "binding_assay"}
            )
        )
        result = dbase_query(manager, {"limit": 1})
        assert result["count"] == 1
        assert len(result["records"]) == 1
    finally:
        tmp.cleanup()


def test_dbase_query_filters_by_y_type():
    manager, tmp = _new_manager_with_stub()
    try:
        manager.write_record(
            _make_evidence(
                y={"type": "logP", "category": "pk_adme", "value": [3.5], "assay": "binding_assay"}
            )
        )
        manager.write_record(
            _make_evidence(
                y={"type": "ic50", "category": "binding", "value": [5.0], "assay": "binding_assay"}
            )
        )
        result = dbase_query(manager, {"y_type": "logP"})
        assert result["count"] >= 1
        assert all(r.get("y", {}).get("type") == "logP" for r in result["records"])
    finally:
        tmp.cleanup()


def test_dbase_query_empty_filter_returns_all():
    manager, tmp = _new_manager_with_stub()
    try:
        dbase_write(manager, _make_evidence())
        dbase_write(
            manager,
            _make_evidence(
                y={"type": "ic50", "category": "binding", "value": [5.0], "assay": "binding_assay"},
                biological_level="cellular",
                cell_line_id="cellosaurus:CVCL_0001",
            ),
        )
        result = dbase_query(manager, {})
        assert result["count"] >= 2
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# dbase_write
# ---------------------------------------------------------------------------


def test_dbase_write_returns_correct_structure():
    manager, tmp = _new_manager_with_stub()
    try:
        record = _make_evidence()
        result = dbase_write(manager, record)
        assert isinstance(result, dict)
        assert "evidence_id" in result
        assert "written" in result
        assert "embedding_generated" in result
        assert result["written"] is True
        assert result["embedding_generated"] is True
    finally:
        tmp.cleanup()


def test_dbase_write_generates_embedding():
    manager, tmp = _new_manager_with_stub()
    try:
        record = _make_evidence()
        result = dbase_write(manager, record)
        assert result["embedding_generated"] is True
        assert result["evidence_id"].startswith("ev_")
    finally:
        tmp.cleanup()


def test_dbase_write_rejects_invalid_record():
    manager, tmp = _new_manager_with_stub()
    try:
        with pytest.raises(ValueError):
            dbase_write(manager, _make_evidence(biological_level="invalid_level"))
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# dbase_status
# ---------------------------------------------------------------------------


def test_dbase_status_returns_expected_keys():
    manager, tmp = _new_manager_with_stub()
    try:
        manager.write_record(_make_evidence())
        result = dbase_status(manager)
        assert isinstance(result, dict)
        assert "record_count" in result
        assert "shard_count" in result
        assert "has_parquet_view" in result
        assert isinstance(result["record_count"], int)
        assert isinstance(result["shard_count"], int)
        assert isinstance(result["has_parquet_view"], bool)
    finally:
        tmp.cleanup()


def test_dbase_status_reflects_written_records():
    manager, tmp = _new_manager_with_stub()
    try:
        result0 = dbase_status(manager)
        assert result0["record_count"] == 0
        manager.write_record(_make_evidence())
        result1 = dbase_status(manager)
        assert result1["record_count"] == 1
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# DBaseStore with injected EmbeddingModel
# ---------------------------------------------------------------------------


def test_manager_with_injected_embedding_model():
    backend = _StubEmbeddingBackend()
    emb_model = EmbeddingModel(backend)
    tmp = tempfile.TemporaryDirectory()
    try:
        dbase = DBase("test", root_dir=tmp.name)
        manager = DBaseStore(dbase, embedding_model=emb_model)
        vec = manager._embed("hello")
        assert len(vec) == 384
        # Stub produces distinct vectors per text — verify we got a non-zero vector
        assert not np.allclose(vec, np.zeros(384, dtype=np.float32))
    finally:
        tmp.cleanup()


def test_manager_without_injection_uses_lazy_default():
    """Without injection, _embedding_model starts as None and gets lazily created.

    The lazy default creates a real SentenceTransformer model, which may or may
    not be available, so we only check that the manager tracks the model field
    correctly without triggering the lazy default here.
    """
    tmp = tempfile.TemporaryDirectory()
    try:
        dbase = DBase("test", root_dir=tmp.name)
        manager = DBaseStore(dbase)
        assert manager._embedding_model is None
    finally:
        tmp.cleanup()


def test_manager_passing_none_embedding_model_uses_default():
    """DBaseStore with explicit None should initialize _embedding_model to None."""
    tmp = tempfile.TemporaryDirectory()
    try:
        dbase = DBase("test", root_dir=tmp.name)
        manager = DBaseStore(dbase, embedding_model=None)
        assert manager._embedding_model is None
    finally:
        tmp.cleanup()


# ---------------------------------------------------------------------------
# DBaseStore without injected embedding model
# ---------------------------------------------------------------------------


def test_manager_init_without_embedding_model_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseStore(dbase)
        # Constructor with DBase only must still work; embedding is lazy
        assert manager.dbase is not None
        assert manager._embedding_model is None
