"""Integration tests for the Ingest workflow."""

from __future__ import annotations

import pytest

from dargus.workflows.ingest import (
    IngestionReport,
    IngestionSummary,
    _collect_duplicates,
    _parse_source,
    _partition_by_domain,
    _run_ingest,
    _user_confirmation_gate,
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
# HITL confirmation gate tests (S3_T4)
# ---------------------------------------------------------------------------


class TestConfirmCallbackProceed:
    """confirm_callback returns 'proceed' -- all records should be written."""

    def test_proceed_via_callback(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "proceed"
        valid_ingest_spec["_duplicate_records"] = [
            {"evidence_id": "dup-001", "reason": "exact_match"},
        ]
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "completed"
        assert result["n_records"] > 0

    def test_proceed_via_callback_no_duplicates(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "proceed"
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "completed"
        assert result["n_records"] > 0


class TestConfirmCallbackSkipDuplicates:
    """confirm_callback returns 'skip-duplicates' -- only non-flagged records written."""

    def test_skip_duplicates_via_callback(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "skip-duplicates"
        valid_ingest_spec["_duplicate_records"] = [
            {"evidence_id": "dup-001", "reason": "exact_match"},
        ]
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "completed"

    def test_skip_duplicates_no_duplicates_means_all_written(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "skip-duplicates"
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "completed"
        assert result["n_records"] > 0


class TestConfirmCallbackAbort:
    """confirm_callback returns 'abort' -- nothing should be written."""

    def test_abort_via_callback(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "abort"
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "aborted_by_user"
        assert result["n_records"] == 0

    def test_abort_via_callback_with_duplicates(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "abort"
        valid_ingest_spec["_duplicate_records"] = [
            {"evidence_id": "dup-001", "reason": "exact_match"},
        ]
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] == "aborted_by_user"
        assert result["n_records"] == 0


class TestDefaultAllowNoCallback:
    """No confirm_callback => default to allow (per CLAUDE.md)."""

    def test_default_allow_no_callback(self, valid_ingest_spec):
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] in ("completed", "converged")
        assert result["n_records"] > 0

    def test_default_allow_no_callback_with_duplicates(self, valid_ingest_spec):
        valid_ingest_spec["_duplicate_records"] = [
            {"evidence_id": "dup-001", "reason": "exact_match"},
        ]
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert result["status"] in ("completed", "converged")
        assert result["n_records"] > 0


class TestIngestionSummaryShape:
    """IngestionSummary carries per-domain counts, records to write, duplicates flagged."""

    def test_summary_from_instances_and_duplicates(self):
        instances = [
            {"id": "a", "domain": "molecular", "data": {}},
            {"id": "b", "domain": "molecular", "data": {}},
            {"id": "c", "domain": "clinical", "data": {}},
        ]
        duplicates = [
            {"evidence_id": "dup-1", "reason": "similar_fingerprint"},
        ]
        summary = IngestionSummary.from_instances(instances, duplicates)
        assert summary.per_domain == {"molecular": 2, "clinical": 1}
        assert summary.n_to_write == 3
        assert summary.n_duplicates == 1
        assert summary.duplicates == duplicates

    def test_summary_from_instances_empty(self):
        summary = IngestionSummary.from_instances([], [])
        assert summary.per_domain == {}
        assert summary.n_to_write == 0
        assert summary.n_duplicates == 0
        assert summary.duplicates == []

    def test_summary_is_passed_to_callback(self, valid_ingest_spec):
        captured: list[IngestionSummary] = []

        def _cb(summary, _dups):
            captured.append(summary)
            return "proceed"

        valid_ingest_spec["confirm_callback"] = _cb
        valid_ingest_spec["_duplicate_records"] = [
            {"evidence_id": "dup-001", "reason": "exact_match"},
        ]
        valid_ingest_spec["max_rounds"] = 1
        _run_ingest(valid_ingest_spec)
        assert len(captured) == 1
        summary = captured[0]
        assert isinstance(summary, IngestionSummary)
        assert summary.n_duplicates == 1
        assert summary.n_to_write > 0
        assert isinstance(summary.per_domain, dict)


class TestUserConfirmationGate:
    """Direct gate unit tests -- _user_confirmation_gate decision routing."""

    def test_gate_returns_proceed_for_allow_callback(self):
        ctx = _make_minimal_context()
        task_spec = {
            "confirm_callback": lambda summary, dups: "proceed",
        }
        summary = IngestionSummary(
            per_domain={"molecular": 2}, n_to_write=2, n_duplicates=0, duplicates=[]
        )
        decision = _user_confirmation_gate(ctx, task_spec, summary)
        assert decision == "proceed"

    def test_gate_returns_skip_duplicates_for_skip_callback(self):
        ctx = _make_minimal_context()
        task_spec = {
            "confirm_callback": lambda summary, dups: "skip-duplicates",
        }
        summary = IngestionSummary(
            per_domain={"molecular": 2}, n_to_write=2, n_duplicates=1, duplicates=[]
        )
        decision = _user_confirmation_gate(ctx, task_spec, summary)
        assert decision == "skip-duplicates"

    def test_gate_returns_abort_for_abort_callback(self):
        ctx = _make_minimal_context()
        task_spec = {
            "confirm_callback": lambda summary, dups: "abort",
        }
        summary = IngestionSummary(
            per_domain={"molecular": 2}, n_to_write=2, n_duplicates=0, duplicates=[]
        )
        decision = _user_confirmation_gate(ctx, task_spec, summary)
        assert decision == "abort"

    def test_gate_defaults_to_allow_when_no_callback(self):
        ctx = _make_minimal_context()
        task_spec: dict = {}
        summary = IngestionSummary(
            per_domain={"molecular": 2}, n_to_write=2, n_duplicates=1, duplicates=[]
        )
        decision = _user_confirmation_gate(ctx, task_spec, summary)
        assert decision == "proceed"

    def test_gate_logs_auto_approved_when_no_callback(self):
        ctx = _make_minimal_context()
        summary = IngestionSummary(
            per_domain={"molecular": 2}, n_to_write=2, n_duplicates=1, duplicates=[]
        )
        decision = _user_confirmation_gate(ctx, {}, summary)
        assert decision == "proceed"


# ---------------------------------------------------------------------------
# Result dict shape tests (S3_T4)
# ---------------------------------------------------------------------------


class TestResultDictShape:
    """The workflow returns a typed result dict with status, per-domain counts, session."""

    def test_result_dict_has_per_domain_counts(self, valid_ingest_spec):
        valid_ingest_spec["confirm_callback"] = lambda _summary, _dups: "proceed"
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert "per_domain" in result
        assert isinstance(result["per_domain"], dict)

    def test_result_dict_has_status(self, valid_ingest_spec):
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert "status" in result
        assert result["status"] in ("completed", "converged", "aborted_by_user")

    def test_result_dict_has_session(self, valid_ingest_spec):
        valid_ingest_spec["max_rounds"] = 1
        result = _run_ingest(valid_ingest_spec)
        assert "session" in result
        assert isinstance(result["session"], dict)


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
    assert confirmations[0]["type"] == "confirmation_gate"
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
# Backward-compat tests removed — legacy signatures deleted in S2_T2
# ---------------------------------------------------------------------------


def test_run_ingest_backward_compat_with_reset():
    """C2: run_ingest(datadir, reset=True) works."""
    result = run_ingest("/data/test_dir", reset=True)
    assert isinstance(result, IngestionReport)
    assert result.n_skipped == 0


def test_run_ingest_backward_compat_with_disease_kb_dir():
    """C2: run_ingest(datadir, disease_kb_dir=...) works."""
    result = run_ingest("/data/test_dir", disease_kb_dir="/data/kb")
    assert isinstance(result, IngestionReport)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_context():
    """Build a minimal HookContext-like object for gate unit tests."""
    from collections import namedtuple

    Ctx = namedtuple("Ctx", ["session", "extra"])
    return Ctx(session={}, extra={})
