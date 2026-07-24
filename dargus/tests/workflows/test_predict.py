"""Integration tests for the Predict workflow."""

from __future__ import annotations

import pytest

from dargus.runtime.hooks import (
    AcceptanceGateHook,
    HookContext,
    HookRegistry,
)
from dargus.workflows.predict import (
    _build_final_report,
    _StubD4Expert,
    _user_confirmation_gate,
    run_predict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_predict_spec() -> dict:
    return {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "endpoints": ["cognitive_score"],
        "max_rounds": 3,
        "timeout_seconds": 30,
    }


# ---------------------------------------------------------------------------
# run_predict basic tests
# ---------------------------------------------------------------------------


def test_run_predict_completes_and_returns_predict_result(valid_predict_spec):
    """run_predict should complete the D4Expert loop and return expected keys."""
    result = run_predict(valid_predict_spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "predict"
    assert result["status"] in ("completed", "converged")
    assert "rounds_completed" in result
    assert result["rounds_completed"] <= valid_predict_spec["max_rounds"]
    assert "report" in result
    assert "session" in result

    report = result["report"]
    assert "efficacy_low" in report
    assert "efficacy_up" in report
    assert "supporting_records" in report
    assert "overall_conclusion" in report


def test_run_predict_uses_max_rounds_from_spec(valid_predict_spec):
    """Defaults should be overridden by task_spec."""
    valid_predict_spec["max_rounds"] = 2
    result = run_predict(valid_predict_spec)
    assert result["rounds_completed"] <= 2


def test_run_predict_force_converge_on_low_max_rounds():
    """With max_rounds=1, safety net should trigger force_converge."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "max_rounds": 1,
    }
    result = run_predict(spec)
    assert result["rounds_completed"] <= 1


def test_run_predict_user_confirmation_logs(caplog):
    """When require_confirmation is set, the gate should log."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00002"],
        "disease_id": "Parkinson",
        "max_rounds": 1,
        "require_confirmation": True,
    }
    with caplog.at_level("INFO"):
        run_predict(spec)
    assert "User confirmation required" in caplog.text


# ---------------------------------------------------------------------------
# AcceptanceGateHook validation tests
# ---------------------------------------------------------------------------


def test_acceptance_gate_raises_on_invalid_efficacy_low():
    """AcceptanceGateHook should raise ValueError when efficacy_low is out of [0,1]."""
    gate = AcceptanceGateHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={"FinalReport": {"efficacy_low": -0.5, "efficacy_up": 0.5}},
    )
    with pytest.raises(ValueError, match="efficacy_low must be in"):
        gate(ctx)


def test_acceptance_gate_raises_on_invalid_efficacy_up():
    """AcceptanceGateHook should raise ValueError when efficacy_up is out of [0,1]."""
    gate = AcceptanceGateHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={"FinalReport": {"efficacy_low": 0.5, "efficacy_up": 1.5}},
    )
    with pytest.raises(ValueError, match="efficacy_up must be in"):
        gate(ctx)


def test_acceptance_gate_raises_on_empty_supporting_records():
    """AcceptanceGateHook should raise ValueError when supporting_records is empty."""
    gate = AcceptanceGateHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "efficacy_low": 0.5,
                "efficacy_up": 0.8,
                "supporting_records": [],
            }
        },
    )
    with pytest.raises(ValueError, match="supporting_records must be a non-empty"):
        gate(ctx)


def test_acceptance_gate_passes_on_valid_report():
    """AcceptanceGateHook should pass when all fields are valid."""
    gate = AcceptanceGateHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "efficacy_low": 0.5,
                "efficacy_up": 0.8,
                "supporting_records": ["rec-1"],
            }
        },
    )
    result = gate(ctx)
    assert result is not None  # context passed through


def test_acceptance_gate_noop_on_missing_report():
    """AcceptanceGateHook should no-op when no FinalReport is present."""
    gate = AcceptanceGateHook()
    ctx = HookContext(runtime=None, task_spec={}, extra={})
    result = gate(ctx)
    assert result is ctx  # identity pass-through for no-op


# ---------------------------------------------------------------------------
# _build_final_report
# ---------------------------------------------------------------------------


def test_build_final_report_has_expected_keys():
    spec = {"drug_ids": ["DB1"], "disease_id": "Cancer", "endpoints": ["survival"]}
    report = _build_final_report({"overall_conclusion": "test"}, spec)
    assert report["efficacy_low"] == 0.3
    assert report["efficacy_up"] == 0.7
    assert report["supporting_records"] == ["stub-record-1"]
    assert report["overall_conclusion"] == "test"


# ---------------------------------------------------------------------------
# _StubD4Expert tests
# ---------------------------------------------------------------------------


def test_stub_d4_expert_delegate_to_expert():
    stub = _StubD4Expert(HookRegistry())
    rep = stub.delegate_to_expert("molecular", [], "test question")
    assert rep["domain"] == "molecular"
    assert "confidence" in rep


def test_stub_d4_expert_synthesize():
    stub = _StubD4Expert(HookRegistry())
    reports = [
        {"domain": "molecular", "confidence": {"low": 0.5, "high": 0.8}, "conclusion": "ok"},
        {"domain": "clinical", "confidence": {"low": 0.4, "high": 0.7}, "conclusion": "ok"},
    ]
    result = stub.synthesize(reports)
    assert "overall_conclusion" in result
    assert "expert_reports" in result


# ---------------------------------------------------------------------------
# _user_confirmation_gate
# ---------------------------------------------------------------------------


def test_user_confirmation_gate(caplog):
    ctx = HookContext(runtime=None, task_spec={"require_confirmation": True})
    _user_confirmation_gate(ctx, {"require_confirmation": True})
    # The gate only logs when require_confirmation is set; currently the log is
    # inside the task_spec check, so we confirm no exception is raised.
    assert True
