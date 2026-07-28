"""Tests for P1.6: ToolCache, embedding tool, routing skill, new dbase tools."""

from __future__ import annotations

import tempfile

import pytest

from dargus.dbase import DBase
from dargus.dbase.store import DBaseStore
from dargus.models.embedding import Embedding, EmbeddingBackend, EmbeddingModel
from dargus.tools.base import Tool
from dargus.tools.cache import ToolCache
from dargus.tools.dbase import (
    dbase_update_status,
    dbase_write_summary,
    make_dbase_tools,
    register_dbase_tools,
)
from dargus.tools.embedding import embedding
from dargus.tools.registry import ToolRegistry


class _StubEmbeddingBackend(EmbeddingBackend):
    """Deterministic per-text vectors for testing."""

    _model_name = "stub-embedder-v1"

    def embed(self, texts: list[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for text in texts:
            seed = abs(hash(text)) % 10000
            out.append([0.01 * ((seed + i) % 100) for i in range(384)])
        return out


def _make_evidence(**overrides):
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {"type": "logP", "category": "pk_adme", "value": [3.5]},
        "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


def _new_manager(tmp):
    dbase = DBase("test", root_dir=tmp)
    emb = EmbeddingModel(_StubEmbeddingBackend())
    return DBaseStore(dbase, embedding_model=emb), dbase


# ---------------------------------------------------------------------------
# ToolCache
# ---------------------------------------------------------------------------


def test_toolcache_lazy_factory_called_once():
    cache = ToolCache()
    calls = []
    resource = cache.get("model", lambda: calls.append(1) or object())
    again = cache.get("model")
    assert resource is again
    assert len(calls) == 1


def test_toolcache_missing_key_raises():
    cache = ToolCache()
    with pytest.raises(KeyError):
        cache.get("nope")


def test_toolcache_close_runs_closers_and_blocks_get():
    cache = ToolCache()
    closed = []
    cache.put("heavy", "resource", closer=lambda r: closed.append(r))
    cache.close()
    assert closed == ["resource"]
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("heavy")
    cache.close()  # idempotent


# ---------------------------------------------------------------------------
# embedding tool
# ---------------------------------------------------------------------------


def test_embedding_tool_ops_use_cache():
    cache = ToolCache()
    cache.put("embedding_model", EmbeddingModel(_StubEmbeddingBackend()))

    info = embedding(op="info", cache=cache)
    assert info["model_name"] == "stub-embedder-v1"

    result = embedding(texts=["hello", "world"], op="embed", cache=cache)
    assert len(result["vectors"]) == 2
    assert len(result["vectors"][0]) == 384

    probe = embedding(op="test", cache=cache)
    assert probe["ok"] is True
    assert probe["dimension"] == 384

    with pytest.raises(ValueError, match="Unknown embedding op"):
        embedding(op="bogus", cache=cache)


# ---------------------------------------------------------------------------
# new dbase tools
# ---------------------------------------------------------------------------


def test_dbase_update_status_tool():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        manager.write_record(record)
        eid = record["evidence_id"]

        out = dbase_update_status(manager, eid, "holdout-test")
        assert out == {"evidence_id": eid, "status": "holdout-test"}
        assert manager.read_records() == []

        out = dbase_update_status(manager, eid, "active")
        assert out["status"] == "active"

        with pytest.raises(ValueError, match="not in"):
            dbase_update_status(manager, eid, "bogus-status")


def test_dbase_write_summary_tool():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        manager.write_record(record)
        eid = record["evidence_id"]

        out = dbase_write_summary(manager, eid, "Aspirin logP at pH 7.4")
        assert out == {"evidence_id": eid, "written": True}
        assert manager.read_summary(eid) == "Aspirin logP at pH 7.4"


def test_make_dbase_tools_returns_bound_tools():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        tools = make_dbase_tools(manager)
        names = {t.name for t in tools}
        assert names == {
            "dbase_query",
            "dbase_write",
            "dbase_status",
            "dbase_update_status",
            "dbase_write_summary",
        }
        assert all(isinstance(t, Tool) for t in tools)

        # bound impls execute against the manager
        status = next(t for t in tools if t.name == "dbase_status")
        assert status.execute()["record_count"] == 0

        record = _make_evidence()
        write = next(t for t in tools if t.name == "dbase_write")
        out = write.execute(record=record)
        assert out["written"] is True

        query = next(t for t in tools if t.name == "dbase_query")
        assert query.execute(x_entity="chembl:CHEMBL25")["count"] == 1


def test_register_dbase_tools_replaces_registry_stubs():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        registry = ToolRegistry()
        register_dbase_tools(manager, registry)
        tool = registry.get("dbase_query")
        record = _make_evidence()
        manager.write_record(record)
        assert tool.execute(disease_id="mondo:0005148")["count"] == 1


def test_registry_yaml_declares_new_tools():
    registry = ToolRegistry()
    for name in ("dbase_update_status", "dbase_write_summary", "embedding"):
        tool = registry.get(name)  # raises if missing
        assert tool.name == name


# ---------------------------------------------------------------------------
# routing skill
# ---------------------------------------------------------------------------


def test_route_field_match_and_sidecar_ranking():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        r1 = _make_evidence(source_entry="10.1234/a")
        r2 = _make_evidence(
            y={"type": "solubility", "category": "pk_adme", "value": [-4.1]},
            source_entry="10.1234/b",
        )
        r3 = _make_evidence(
            biological_level="rct",
            y={"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.7]},
            source_entry="10.1234/c",
        )
        for r in (r1, r2, r3):
            assert manager.write_record(r) is True

        # y_type field match restricts the candidate set
        results = manager.route("aspirin logP", y_type="logP")
        assert [r["evidence_id"] for r, _ in results] == [r1["evidence_id"]]

        # biological_level field match
        results = manager.route("clinical response", biological_level="rct")
        assert [r["evidence_id"] for r, _ in results] == [r3["evidence_id"]]

        # disease_id field match — all three carry mondo:0005148
        results = manager.route("anything", disease_id="mondo:0005148")
        assert len(results) == 3
        # every ranked record has a sidecar vector → all scores > 0, sorted desc
        scores = [s for _, s in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)


def test_route_bg_drugs_filter():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        r1 = _make_evidence(source_entry="10.1234/a")
        r2 = _make_evidence(
            x={
                "type": "drug",
                "value": [{"entity_id": "chembl:CHEMBL99", "entity_label": "other"}],
            },
            y={"type": "solubility", "category": "pk_adme", "value": [-4.1]},
            source_entry="10.1234/b",
        )
        manager.write_record(r1)
        manager.write_record(r2)

        results = manager.route("query", bg_drugs=["chembl:CHEMBL99"])
        assert [r["evidence_id"] for r, _ in results] == [r2["evidence_id"]]


def test_route_skips_records_without_sidecar_vector():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        r1 = _make_evidence(source_entry="10.1234/a")
        r2 = _make_evidence(
            y={"type": "solubility", "category": "pk_adme", "value": [-4.1]},
            source_entry="10.1234/b",
        )
        manager.write_record(r1)
        manager.write_record(r2)

        # wipe r2's vector from the active sidecar
        fp = dbase.sidecars.active_fingerprint()
        path = dbase.sidecars.embeddings_path(fp)
        lines = [line for line in path.read_text().splitlines() if r2["evidence_id"] not in line]
        path.write_text("\n".join(lines) + "\n")

        results = manager.route("aspirin logP")
        assert len(results) == 2
        by_id = {r["evidence_id"]: s for r, s in results}
        assert by_id[r2["evidence_id"]] == 0.0  # no vector → score 0, sorts last
        assert results[-1][0]["evidence_id"] == r2["evidence_id"]


def test_route_empty_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        assert manager.route("nothing here", y_type="logP") == []


# ---------------------------------------------------------------------------
# dedup: y_type scope + configurable threshold
# ---------------------------------------------------------------------------


def test_dedup_threshold_configurable():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        emb = EmbeddingModel(_StubEmbeddingBackend())
        manager = DBaseStore(dbase, embedding_model=emb, dedup_threshold=0.99)
        assert manager.dedup_threshold == 0.99


def test_semantic_check_scoped_by_y_type():
    """Dedup candidates are restricted to records with the same y.type."""
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        existing = _make_evidence(
            y={"type": "solubility", "category": "pk_adme", "value": [-4.1]},
            source_entry="10.1234/existing",
        )
        manager.write_record(existing, dedup=False)

        # same drug, different y.type → existing record out of dedup scope
        incoming = _make_evidence(source_entry="10.1234/incoming")
        assert manager._semantic_check(incoming) is None
