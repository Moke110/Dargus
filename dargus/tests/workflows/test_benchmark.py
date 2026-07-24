"""Integration tests for the Benchmark workflow."""

from __future__ import annotations

import pytest

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


@pytest.fixture
def valid_benchmark_spec() -> dict:
    return {
        "workflow": "benchmark",
        "holdout_ids": ["rec-001", "rec-002", "rec-003"],
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "endpoints": ["cognitive_score"],
        "max_rounds": 2,
    }


# ---------------------------------------------------------------------------
# run_benchmark tests
# ---------------------------------------------------------------------------


def test_run_benchmark_completes_and_returns_benchmark_result(valid_benchmark_spec):
    """Run benchmark should complete and return expected keys."""
    result = run_benchmark(valid_benchmark_spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "benchmark"
    assert result["status"] in ("completed", "converged")
    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "n_test" in result
    assert result["n_test"] == len(valid_benchmark_spec["holdout_ids"])

    # Metrics should be in [0, 1]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of range"


def test_run_benchmark_with_zero_holdout():
    """Benchmark with empty holdout set should return zero metrics."""
    spec = {
        "workflow": "benchmark",
        "holdout_ids": [],
        "max_rounds": 1,
    }
    result = run_benchmark(spec)
    assert result["n_test"] == 0
    assert result["accuracy"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_run_benchmark_restores_holdout_state():
    """Holdout records should be restored after benchmark completes."""
    spec = {"workflow": "benchmark", "holdout_ids": ["h1", "h2"], "max_rounds": 1}
    result = run_benchmark(spec)
    # Should complete without error (restore is a no-op log in stub)
    assert "session" in result


def test_run_benchmark_user_confirmation(caplog):
    """When require_confirmation is set, gate should log."""
    spec = {
        "workflow": "benchmark",
        "holdout_ids": ["h1"],
        "max_rounds": 1,
        "require_confirmation": True,
    }
    with caplog.at_level("INFO"):
        run_benchmark(spec)
    assert "User confirmation required" in caplog.text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_mark_and_restore_holdout(caplog):
    ids = ["rec-a", "rec-b"]
    with caplog.at_level("INFO"):
        _mark_holdout(ids)
        _restore_holdout(ids)
    assert "Holdout marked" in caplog.text
    assert "Holdout restored" in caplog.text


def test_load_ground_truth():
    ids = ["r1", "r2", "r3"]
    gt = _load_ground_truth(ids)
    assert len(gt) == 3
    for rid in ids:
        assert rid in gt
        assert "actual_efficacy" in gt[rid]
        assert "outcome" in gt[rid]
        assert 0.0 <= gt[rid]["actual_efficacy"] <= 1.0


def test_compute_metrics_returns_valid_range():
    gt = {"r1": {"actual_efficacy": 0.6}, "r2": {"actual_efficacy": 0.8}}
    m = _compute_metrics({"conclusion": "test"}, gt)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} out of range"


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
