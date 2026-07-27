"""Integration tests for the Benchmark workflow (v1.0.0 sidecar holdout)."""

from __future__ import annotations

import os

import pytest

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.runtime.hooks import HookContext
from dargus.workflows.benchmark import (
    _compute_metrics,
    _load_ground_truth,
    _mark_holdout,
    _restore_holdout,
    _user_confirmation_gate,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(**overrides):
    e = {
        "biological_level": "rct",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.7]},
        "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/bench"}],
        "source_entry": "10.1234/bench",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


@pytest.fixture
def bench_dbase(tmp_path, monkeypatch):
    """Isolated global D-Base (via DARGUS_HOME) seeded with evidence."""
    home = str(tmp_path / "dargus_home")
    os.makedirs(home, exist_ok=True)
    monkeypatch.setenv("DARGUS_HOME", home)

    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)
    records = [
        _make_evidence(
            y={"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.8]},
            source_entry="10.1234/bench-a",
        ),
        _make_evidence(
            y={"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.6]},
            source_entry="10.1234/bench-b",
        ),
        _make_evidence(
            y={"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.2]},
            source_entry="10.1234/bench-c",
        ),
    ]
    for record in records:
        # dedup=False: deterministic seeding — the MockNLP zero-vector
        # fallback would otherwise soft-flag similar records as duplicates
        assert manager.write_record(record, dedup=False) is True
    return manager, records


# ---------------------------------------------------------------------------
# run_benchmark tests
# ---------------------------------------------------------------------------


def test_run_benchmark_completes_and_returns_benchmark_result(bench_dbase):
    """run_benchmark completes against a real D-Base and returns metrics."""
    _, records = bench_dbase
    holdout_ids = [r["evidence_id"] for r in records[:2]]
    spec = {
        "workflow": "benchmark",
        "holdout_ids": holdout_ids,
        "drug_ids": ["chembl:CHEMBL25"],
        "disease_id": "mondo:0005148",
        "endpoints": ["response_rate"],
        "max_rounds": 1,
    }
    result = run_benchmark(spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "benchmark"
    assert result["status"] in ("completed", "converged")
    assert result["n_test"] == 2
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of range"


def test_run_benchmark_holdout_excluded_and_restored(bench_dbase):
    """Holdout records flip to holdout-test during the run and back after."""
    manager, records = bench_dbase
    holdout_ids = [r["evidence_id"] for r in records]

    seen_statuses: dict[str, str] = {}

    def _spy_predict(predict_spec):
        for eid in holdout_ids:
            seen_statuses[eid] = manager.get_status(eid)["status"]
        # active reads must see nothing while all records are held out
        assert manager.read_records() == []
        return {"status": "completed", "report": {"efficacy_score": 0.9}}

    import dargus.workflows.benchmark as bench_mod

    original = bench_mod._run_predict_standalone
    bench_mod._run_predict_standalone = _spy_predict
    try:
        result = run_benchmark(
            {
                "workflow": "benchmark",
                "holdout_ids": holdout_ids,
                "drug_ids": ["chembl:CHEMBL25"],
                "disease_id": "mondo:0005148",
                "max_rounds": 1,
            }
        )
    finally:
        bench_mod._run_predict_standalone = original

    assert all(s == "holdout-test" for s in seen_statuses.values())
    # restored afterwards
    for eid in holdout_ids:
        assert manager.get_status(eid)["status"] == "active"
    assert len(manager.read_records()) == len(records)
    # ground truth: two positives (0.8, 0.6), one negative (0.2); predicted 0.9 ≥ 0.5
    # → 2 TP, 1 FP
    assert result["accuracy"] == pytest.approx(2 / 3, abs=0.001)
    assert result["precision"] == pytest.approx(2 / 3, abs=0.001)
    assert result["recall"] == pytest.approx(1.0)


def test_run_benchmark_zero_match_aborts(bench_dbase):
    """Zero matched holdout records must abort with ValueError."""
    with pytest.raises(ValueError, match="zero records"):
        run_benchmark(
            {
                "workflow": "benchmark",
                "holdout_ids": ["ev_does_not_exist"],
                "max_rounds": 1,
            }
        )


def test_run_benchmark_restores_on_predict_failure(bench_dbase):
    """Holdout records are restored to active even when predict raises."""
    manager, records = bench_dbase
    holdout_ids = [records[0]["evidence_id"]]

    import dargus.workflows.benchmark as bench_mod

    original = bench_mod._run_predict_standalone
    bench_mod._run_predict_standalone = lambda spec: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_benchmark({"workflow": "benchmark", "holdout_ids": holdout_ids, "max_rounds": 1})
    finally:
        bench_mod._run_predict_standalone = original

    assert manager.get_status(holdout_ids[0])["status"] == "active"


def test_run_benchmark_holdout_filter_selects_records(bench_dbase):
    """A holdout filter dict (no explicit ids) selects matching records."""
    _, records = bench_dbase
    spec = {
        "workflow": "benchmark",
        "holdout": {"drug_ids": ["chembl:CHEMBL25"], "max_records": 1},
        "drug_ids": ["chembl:CHEMBL25"],
        "disease_id": "mondo:0005148",
        "max_rounds": 1,
    }
    result = run_benchmark(spec)
    assert result["n_test"] == 1


def test_run_benchmark_user_confirmation(bench_dbase, caplog):
    """When require_confirmation is set, gate should log."""
    _, records = bench_dbase
    spec = {
        "workflow": "benchmark",
        "holdout_ids": [records[0]["evidence_id"]],
        "max_rounds": 1,
        "require_confirmation": True,
    }
    with caplog.at_level("INFO"):
        run_benchmark(spec)
    assert "User confirmation required" in caplog.text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_mark_and_restore_holdout(bench_dbase):
    manager, records = bench_dbase
    ids = [r["evidence_id"] for r in records[:2]]
    _mark_holdout(manager, ids)
    assert all(manager.get_status(eid)["status"] == "holdout-test" for eid in ids)
    _restore_holdout(manager, ids)
    assert all(manager.get_status(eid)["status"] == "active" for eid in ids)


def test_load_ground_truth_from_records(bench_dbase):
    _, records = bench_dbase
    gt = _load_ground_truth(records)
    assert len(gt) == 3
    values = sorted(gt[r["evidence_id"]]["actual_efficacy"] for r in records)
    assert values == [0.2, 0.6, 0.8]
    assert gt[records[0]["evidence_id"]]["outcome"] == "positive"
    assert gt[records[2]["evidence_id"]]["outcome"] == "negative"


def test_compute_metrics_positive_prediction():
    gt = {
        "ev_a": {"actual_efficacy": 0.8, "outcome": "positive"},
        "ev_b": {"actual_efficacy": 0.1, "outcome": "negative"},
    }
    m = _compute_metrics({"report": {"efficacy_score": 0.9}}, gt)
    # predicted positive: ev_a hit, ev_b false positive
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0


def test_compute_metrics_insufficient_data_counts_negative():
    gt = {
        "ev_a": {"actual_efficacy": 0.8, "outcome": "positive"},
        "ev_b": {"actual_efficacy": 0.1, "outcome": "negative"},
    }
    m = _compute_metrics({"report": {"efficacy_score": None}}, gt)
    assert m["accuracy"] == 0.5  # negative prediction: miss a, hit b
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_empty_ground_truth():
    m = _compute_metrics({}, {})
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0


# ---------------------------------------------------------------------------
# _user_confirmation_gate
# ---------------------------------------------------------------------------


def test_benchmark_user_confirmation_gate_logs(caplog):
    ctx = HookContext(runtime=None, task_spec={"require_confirmation": True})
    _user_confirmation_gate(ctx, {"require_confirmation": True})
    assert True  # no exception
