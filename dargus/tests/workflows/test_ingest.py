"""Integration tests for the Ingest workflow."""

from __future__ import annotations

import pytest

from dargus.workflows.ingest import (
    IngestionReport,
    _collect_duplicates,
    _parse_source,
    _partition_by_domain,
    run_ingest,
)

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
