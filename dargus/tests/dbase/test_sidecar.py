"""Tests for D-Base v1.0.0 sidecar storage (status, llm_summary, embeddings)."""

import json
import tempfile

from dargus.dbase import DBase
from dargus.dbase.sidecar import SidecarStore, model_fingerprint
from dargus.dbase.store import DBaseStore
from dargus.models.embedding import Embedding, EmbeddingBackend, EmbeddingModel


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
        "bg": {"disease_id": [], "drugs": [], "genes": []},
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


# ── SidecarStore primitives ──────────────────────────────────────────────────


def test_status_default_active():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        assert store.read_status("ev_missing") == {"status": "active", "superseded_by": None}


def test_status_latest_wins():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        store.append_status("ev_1", "holdout-test")
        store.append_status("ev_1", "active")
        assert store.read_status("ev_1")["status"] == "active"


def test_status_superseded_by():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        store.append_status("ev_old", "superseded", superseded_by="ev_new")
        st = store.read_status("ev_old")
        assert st["status"] == "superseded"
        assert st["superseded_by"] == "ev_new"


def test_status_invalid_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        try:
            store.append_status("ev_1", "bogus")
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_summary_latest_wins():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        store.append_summary("ev_1", "first")
        store.append_summary("ev_1", "second")
        assert store.read_summary("ev_1") == "second"
        assert store.read_summary("ev_none") is None


def test_embeddings_per_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        store.append_embedding("ev_1", [0.1, 0.2], "fpAAA")
        store.append_embedding("ev_1", [0.9, 0.9], "fpBBB")
        assert store.read_embeddings("fpAAA") == {"ev_1": [0.1, 0.2]}
        assert store.read_embeddings("fpBBB") == {"ev_1": [0.9, 0.9]}
        manifest = store.read_embeddings_manifest()
        assert set(manifest["available"]) == {"fpAAA", "fpBBB"}
        assert manifest["active"] == "fpAAA"  # first registered becomes active


def test_embeddings_manifest_switch_active():
    with tempfile.TemporaryDirectory() as tmp:
        store = SidecarStore(tmp)
        store.append_embedding("ev_1", [0.1], "fpAAA")
        store.set_active_fingerprint("fpBBB")
        assert store.active_fingerprint() == "fpBBB"
        store.set_active_fingerprint("fpAAA")  # switch back, no recomputation
        assert store.active_fingerprint() == "fpAAA"


def test_fingerprint_stable():
    assert model_fingerprint("all-MiniLM-L6-v2") == model_fingerprint("all-MiniLM-L6-v2")
    assert model_fingerprint("a") != model_fingerprint("b")


# ── manager write path: embedding goes to sidecar, not record ────────────────


def test_write_record_stores_embedding_in_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        record = _make_evidence()
        assert manager.write_record(record) is True
        # record itself carries no embedding
        stored = dbase.read_shards()[0]
        assert "embedding" not in stored
        # sidecar has the vector keyed by evidence_id under the model fingerprint
        fp = model_fingerprint("stub-embedder-v1")
        vectors = dbase.sidecars.read_embeddings(fp)
        assert stored["evidence_id"] in vectors
        assert len(vectors[stored["evidence_id"]]) == 384
        assert dbase.sidecars.active_fingerprint() == fp


def test_write_without_embedding_model_still_writes():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseStore(dbase)  # no embedding model
        record = _make_evidence()
        assert manager.write_record(record) is True
        assert len(dbase.read_shards()) == 1


# ── status lifecycle via manager ─────────────────────────────────────────────


def test_supersede_flow():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        old = _make_evidence()
        assert manager.write_record(old) is True
        new = _make_evidence(y={"type": "logP_v2", "category": "pk_adme", "value": [3.6]})
        assert manager.supersede(old["evidence_id"], new) is True
        st = manager.get_status(old["evidence_id"])
        assert st["status"] == "superseded"
        assert st["superseded_by"] == new["evidence_id"]
        # superseded record hidden from default (active) reads
        active = manager.read_records()
        assert all(r["evidence_id"] != old["evidence_id"] for r in active)
        # but visible with status=None
        everything = manager.read_records(status=None)
        assert any(r["evidence_id"] == old["evidence_id"] for r in everything)


def test_retract_hides_from_active_reads():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        manager.write_record(record)
        manager.retract(record["evidence_id"])
        assert manager.read_records() == []
        assert manager.get_status(record["evidence_id"])["status"] == "retracted"


def test_holdout_flip_and_restore():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        manager.write_record(record)
        eid = record["evidence_id"]
        manager.update_status(eid, "holdout-test")
        assert manager.read_records() == []  # invisible to Predict
        assert len(manager.read_records(status="holdout-test")) == 1
        manager.update_status(eid, "active")
        assert len(manager.read_records()) == 1


def test_write_summary_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        manager.write_record(record)
        manager.write_summary(record["evidence_id"], "Aspirin logP measured at pH 7.4.")
        assert manager.read_summary(record["evidence_id"]) == "Aspirin logP measured at pH 7.4."


# ── re-embedding ─────────────────────────────────────────────────────────────


def test_reembed_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        manager.write_record(_make_evidence())
        result1 = manager.reembed()
        assert result1 == {"written": 0, "skipped": 1}  # already embedded at write
        # wipe the sidecar and re-embed from scratch
        fp = dbase.sidecars.active_fingerprint()
        dbase.sidecars.embeddings_path(fp).unlink()
        result2 = manager.reembed()
        assert result2 == {"written": 1, "skipped": 0}
        result3 = manager.reembed()
        assert result3 == {"written": 0, "skipped": 1}


def test_reembed_skips_non_active_records():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        manager.write_record(_make_evidence())
        fp = dbase.sidecars.active_fingerprint()
        dbase.sidecars.embeddings_path(fp).unlink()
        record2 = _make_evidence(y={"type": "solubility", "category": "pk_adme", "value": [-4.1]})
        manager.write_record(record2)
        manager.retract(record2["evidence_id"])
        dbase.sidecars.embeddings_path(fp).unlink()
        result = manager.reembed()
        assert result["written"] == 1  # only the active record
        vectors = dbase.sidecars.read_embeddings(fp)
        assert record2["evidence_id"] not in vectors


def test_sidecar_files_never_in_identity():
    """Status/summary/embedding sidecars don't change evidence_id."""
    from dargus.dbase.validate import compute_evidence_id

    with tempfile.TemporaryDirectory() as tmp:
        manager, _ = _new_manager(tmp)
        record = _make_evidence()
        id_before = compute_evidence_id(record)
        manager.write_record(record)
        manager.update_status(record["evidence_id"], "holdout-test")
        manager.write_summary(record["evidence_id"], "some summary")
        id_after = compute_evidence_id(record)
        assert id_before == id_after


def test_sidecar_files_created_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        manager, dbase = _new_manager(tmp)
        manager.write_record(_make_evidence())
        sidecars = dbase.dbase_dir / "sidecars"
        assert sidecars.is_dir()
        fp = dbase.sidecars.active_fingerprint()
        assert (sidecars / f"embeddings-{fp}.jsonl").exists()
        manifest = json.loads((sidecars / "embeddings_manifest.json").read_text())
        assert manifest["active"] == fp
