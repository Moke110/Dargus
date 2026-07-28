"""Integration tests for the Ingest workflow."""

from __future__ import annotations

import pytest

from dargus.dbase import DBase
from dargus.dbase.store import DBaseStore
from dargus.models.embedding import Embedding, EmbeddingBackend, EmbeddingModel
from dargus.workflows.ingest import (
    IngestionReport,
    _collect_duplicates,
    _parse_source,
    _partition_by_domain,
    _run_ingest,
    run_ingest,
)

# ── helpers for Input phase tests ────────────────────────────────────────────


class _StubEmbeddingBackend(EmbeddingBackend):
    """Deterministic per-char embedding for Input phase tests."""

    _model_name = "stub-input-embedder-v1"

    def embed(self, texts: list[str]) -> list[Embedding]:
        import numpy as np

        out: list[Embedding] = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            for i, b in enumerate(text.encode("utf-8")):
                vec[i % 384] += float(b)
            norm = np.linalg.norm(vec)
            out.append((vec / norm if norm else vec).tolist())
        return out


def _make_valid_evidence(**overrides):
    """Return a valid v1.0.0 three-axis evidence dict."""
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
        },
        "bg": {"disease_id": [], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


def _make_invalid_evidence():
    """Return an evidence dict that will fail validation (invalid biological_level)."""
    return _make_valid_evidence(biological_level="invalid_level")


def _make_near_duplicate_of(existing: dict, **overrides):
    """Return an evidence dict that differs slightly from *existing* (near-duplicate)."""
    dup = dict(existing)
    dup["source_entry"] = "10.1234/near-dup"
    dup["source_time"] = "2026-06-01"
    dup["y"] = dict(existing["y"])
    if overrides:
        dup.update(overrides)
    return dup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_ingest_spec() -> dict:
    return {
        "workflow": "ingest",
        "source_path": "/data/pubmed_batch_01",
        "source_type": "pubmed",
        "max_rounds": 5,
    }


# ---------------------------------------------------------------------------
# S3_T3 — Input phase end-to-end tests
# ---------------------------------------------------------------------------


class TestInputPhaseEndToEnd:
    """End-to-end tests for the Input phase: validate, dedup, embed, write."""

    @pytest.fixture
    def _dbase_store(self, tmp_path):
        """Set up a real temp D-Base with a fake embedding model."""
        dbase = DBase("test", root_dir=str(tmp_path))
        emb_model = EmbeddingModel(_StubEmbeddingBackend())
        store = DBaseStore(dbase, embedding_model=emb_model)
        return store

    def test_valid_records_written_through_single_writer(self, _dbase_store):
        """Valid extracted instances are written through DBaseStore.write_record."""
        evidence1 = _make_valid_evidence()
        evidence2 = _make_valid_evidence(
            x={
                "type": "drug",
                "value": [{"entity_id": "chembl:CHEMBL941", "entity_label": "ibuprofen"}],
            },
            y={"type": "logD", "category": "pk_adme", "value": [2.0]},
            source_entry="10.1234/test2",
        )

        task_spec: dict = {
            "workflow": "ingest",
            "source_path": "",
            "max_rounds": 1,
            "_dbase_store": _dbase_store,
            "_extracted_instances": [evidence1, evidence2],
        }
        result = _run_ingest(task_spec)

        assert result["n_records"] == 2
        assert result["n_errors"] == 0
        assert result["n_duplicates"] == 0
        assert result["status"] in ("completed", "converged")

        # Verify records landed in D-Base via the single-writer path
        records = _dbase_store.read_records()
        assert len(records) == 2
        eids = {r["evidence_id"] for r in records}
        assert len(eids) == 2
        for eid in eids:
            assert eid.startswith("ev_")

        # Verify embeddings landed in the sidecar
        fp = _dbase_store._get_embedding_model().model_name
        from dargus.dbase.sidecar import model_fingerprint

        fp_hash = model_fingerprint(fp)
        vectors = _dbase_store.dbase.sidecars.read_embeddings(fp_hash)
        for eid in eids:
            assert eid in vectors, f"embedding missing for {eid}"

    def test_validation_rejects_skipped_and_logged(self, _dbase_store, caplog):
        """Invalid records fail validation, are logged, and skipped — not fatal."""
        valid = _make_valid_evidence()
        invalid = _make_invalid_evidence()

        task_spec: dict = {
            "workflow": "ingest",
            "source_path": "",
            "max_rounds": 1,
            "_dbase_store": _dbase_store,
            "_extracted_instances": [valid, invalid],
        }
        result = _run_ingest(task_spec)

        # Only the valid record should be written
        assert result["n_records"] == 1
        # One validation error should be recorded
        assert result["n_errors"] == 1

        # The valid one landed
        records = _dbase_store.read_records()
        assert len(records) == 1

        # The invalid was logged
        assert (
            "validation" in caplog.text.lower()
            or "invalid" in caplog.text.lower()
            or "error" in caplog.text.lower()
        )

    def test_exact_duplicates_skipped_on_reingest(self, _dbase_store):
        """Writing the exact same evidence twice skips the second."""
        evidence = _make_valid_evidence()

        task_spec: dict = {
            "workflow": "ingest",
            "source_path": "",
            "max_rounds": 1,
            "_dbase_store": _dbase_store,
            "_extracted_instances": [evidence],
        }
        result1 = _run_ingest(task_spec)
        assert result1["n_records"] == 1

        # Re-ingest the same record
        result2 = _run_ingest(task_spec)
        # The record is the exact same evidence_id, so it is skipped
        assert result2["n_records"] == 0
        assert result2["n_duplicates"] == 0  # exact dup not counted as a "duplicate review request"

        # Only one record in D-Base
        records = _dbase_store.read_records()
        assert len(records) == 1

    def test_near_duplicates_surface_as_review_requests(self, _dbase_store):
        """Near-duplicate write results (DuplicateReviewRequest) are surfaced.

        The Input phase captures DuplicateReviewRequest returns from
        write_record and reports them as duplicate review requests.
        """
        from dargus.dbase.store import DuplicateReviewRequest

        evidence = _make_valid_evidence()

        # Write the first record normally
        _dbase_store.write_record(evidence)

        # Create a near-duplicate that differs enough to produce a different
        # evidence_id (so exact dedup does not catch it).
        evidence2 = _make_near_duplicate_of(evidence)
        evidence2["y"] = {**evidence2["y"], "type": "nearLogP"}

        # Patch write_record so the first call goes through normally
        # (it writes evidence2) and returns a DuplicateReviewRequest.
        # This simulates what happens when the store's semantic dedup
        # detects a near-duplicate.
        _orig_write = _dbase_store.write_record

        def _write_then_flag(record, dedup=True):
            # Write through the real store so the record is persisted
            result = _orig_write(record, dedup=dedup)
            if result is True:
                # After successful write, flag it as a near-duplicate of evidence
                return DuplicateReviewRequest(
                    incoming_raw=record,
                    incoming_evidence=record,
                    candidate_evidence=evidence,
                    similarity_score=0.92,
                    candidate_evidence_id=evidence.get("evidence_id", ""),
                )
            return result

        _dbase_store.write_record = _write_then_flag

        task_spec: dict = {
            "workflow": "ingest",
            "source_path": "",
            "max_rounds": 1,
            "_dbase_store": _dbase_store,
            "_extracted_instances": [evidence2],
        }
        result = _run_ingest(task_spec)

        # Both records in D-Base
        records = _dbase_store.read_records()
        assert len(records) == 2  # original + near-dup (both persisted)

        # A duplicate review request surfaced
        assert result["n_duplicates"] == 1

        # Confirmation gate appended session entry
        confirmations = result["session"].get("confirmations", [])
        assert len(confirmations) >= 1
        assert confirmations[0]["type"] == "duplicate_review"

    def test_empty_extracted_list_handled_gracefully(self, _dbase_store):
        """An empty extracted instances list yields zero records and no crash."""
        task_spec: dict = {
            "workflow": "ingest",
            "source_path": "",
            "max_rounds": 1,
            "_dbase_store": _dbase_store,
            "_extracted_instances": [],
        }
        result = _run_ingest(task_spec)
        assert result["n_records"] == 0
        assert result["n_errors"] == 0

    def test_task_spec_fallback_stubs_still_work(self):
        """When no _dbase_store / _extracted_instances are injected, stubs still pass."""
        result = run_ingest({"workflow": "ingest", "source_path": "/data/test", "max_rounds": 1})
        assert result["status"] in ("completed", "converged")


# ---------------------------------------------------------------------------
# run_ingest tests
# ---------------------------------------------------------------------------


def test_run_ingest_completes_and_returns_ingest_result(valid_ingest_spec):
    """Run ingest should complete and return the expected result keys."""
    result = run_ingest(valid_ingest_spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"
    assert result["status"] in ("completed", "converged")
    assert "n_records" in result
    assert "n_duplicates" in result
    assert "n_errors" in result
    assert "session" in result
    assert result["n_records"] > 0  # stub produces records


def test_run_ingest_with_no_source():
    """With empty source_path, should handle gracefully."""
    result = run_ingest({"workflow": "ingest", "source_path": "", "max_rounds": 1})
    assert result["n_records"] == 0
    assert result["n_errors"] == 0


def test_run_ingest_handles_duplicate_review(valid_ingest_spec):
    """Duplicate review gate should store confirmation info."""
    valid_ingest_spec["require_confirmation"] = True
    result = run_ingest(valid_ingest_spec)
    # Stub duplicates are always empty, but the gate still fires the confirmation path
    assert "session" in result


def test_run_ingest_with_max_rounds():
    """With few rounds, should converge early."""
    spec = {"workflow": "ingest", "source_path": "/data/test", "max_rounds": 2}
    result = run_ingest(spec)
    assert result["n_records"] >= 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_parse_source_returns_records():
    records = _parse_source("/data/test")
    assert len(records) == 15  # 5 molecular + 5 biomedical + 5 clinical
    assert all("domain" in r for r in records)
    assert all("id" in r for r in records)


def test_parse_source_empty():
    records = _parse_source("")
    assert records == []


def test_partition_by_domain_groups_correctly():
    records = [
        {"domain": "molecular", "id": "a"},
        {"domain": "molecular", "id": "b"},
        {"domain": "clinical", "id": "c"},
    ]
    groups = _partition_by_domain(records)
    assert len(groups) == 2
    domain_names = {g[0] for g in groups}
    assert domain_names == {"molecular", "clinical"}


def test_collect_duplicates_empty_for_stub():
    records = [{"id": "r1"}, {"id": "r2"}]
    dups = _collect_duplicates(records)
    assert dups == []


def test_collect_duplicates_injectable_via_task_spec():
    """I4: _collect_duplicates returns injected duplicates from task_spec."""
    records = [{"id": "r1"}]
    fake_dups = [
        {"evidence_id": "dup-1", "reason": "similar_fingerprint"},
        {"evidence_id": "dup-2", "reason": "exact_match"},
    ]
    dups = _collect_duplicates(records, task_spec={"_duplicate_records": fake_dups})
    assert len(dups) == 2
    assert dups[0]["evidence_id"] == "dup-1"


def test_ingest_duplicate_review_path_reached(valid_ingest_spec):
    """I4: When duplicates are injected, the n_duplicates > 0 branch is executed."""
    valid_ingest_spec["_duplicate_records"] = [
        {"evidence_id": "dup-001", "reason": "exact_match"},
    ]
    valid_ingest_spec["max_rounds"] = 1
    result = _run_ingest(valid_ingest_spec)
    assert result["n_duplicates"] == 1
    # Confirm the confirmation record was appended to session
    confirmations = result["session"].get("confirmations", [])
    assert len(confirmations) == 1
    assert confirmations[0]["type"] == "duplicate_review"
    assert confirmations[0]["n_duplicates"] == 1


# ---------------------------------------------------------------------------
# Backward-compat dataclasses
# ---------------------------------------------------------------------------


def test_ingestion_report_defaults():
    r = IngestionReport()
    assert r.n_records == 0
    assert r.n_skipped == 0
    assert r.dbase_size == 0
    assert r.errors == []


def test_training_report_is_ingestion_report():
    from dargus.workflows.ingest import TrainingReport

    r = TrainingReport(n_records=10)
    assert isinstance(r, IngestionReport)
    assert r.n_records == 10


# ---------------------------------------------------------------------------
# Backward-compat run_ingest(datadir) signature tests
# ---------------------------------------------------------------------------


def test_run_ingest_backward_compat_datadir_string():
    """C2: run_ingest(datadir) returns IngestionReport for backward compat."""
    result = run_ingest("/data/test_dir")
    assert isinstance(result, IngestionReport)
    assert result.n_records > 0


def test_run_ingest_backward_compat_with_reset():
    """C2: run_ingest(datadir, reset=True) works."""
    result = run_ingest("/data/test_dir", reset=True)
    assert isinstance(result, IngestionReport)
    assert result.n_skipped == 0


def test_run_ingest_backward_compat_with_disease_kb_dir():
    """C2: run_ingest(datadir, disease_kb_dir=...) works."""
    result = run_ingest("/data/test_dir", disease_kb_dir="/data/kb")
    assert isinstance(result, IngestionReport)


def test_run_ingest_new_api_dict():
    """C2: run_ingest(task_spec) still returns dict."""
    result = run_ingest({"workflow": "ingest", "source_path": "/data/test", "max_rounds": 1})
    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"
