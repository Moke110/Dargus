"""Integration tests for the Predict workflow."""

from __future__ import annotations

import pytest

from dargus.runtime.hooks import (
    HookContext,
    ReportValidationHook,
)
from dargus.workflows.predict import (
    _build_final_report,
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
    # The report is the nested contract: {drug_id: {disease_id: {endpoint: {...}}}}
    assert isinstance(report, dict)
    assert "DB00001" in report
    assert "Alzheimer" in report["DB00001"]
    assert "cognitive_score" in report["DB00001"]["Alzheimer"]
    entry = report["DB00001"]["Alzheimer"]["cognitive_score"]
    assert "efficacy_score" in entry
    assert "confidence_score" in entry
    assert "supporting_records" in entry
    assert "reasoning_mode" in entry
    assert "confidence_level" in entry


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
# ReportValidationHook validation tests
# ---------------------------------------------------------------------------


def test_report_validation_raises_on_invalid_efficacy_score():
    """ReportValidationHook should raise when efficacy_score is out of [0,1]."""
    gate = ReportValidationHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={"FinalReport": {"efficacy_score": -0.5, "confidence_score": 0.5}},
    )
    with pytest.raises(ValueError, match="efficacy_score must be in"):
        gate(ctx)


def test_report_validation_raises_on_invalid_confidence_score():
    """ReportValidationHook should raise when confidence_score is out of [0,1]."""
    gate = ReportValidationHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={"FinalReport": {"efficacy_score": 0.5, "confidence_score": 1.5}},
    )
    with pytest.raises(ValueError, match="confidence_score must be in"):
        gate(ctx)


def test_report_validation_waives_scores_on_insufficient_data():
    """insufficient_data reports must have both scores unset — and pass then."""
    gate = ReportValidationHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "confidence_level": "insufficient_data",
                "efficacy_score": None,
                "confidence_score": None,
                "supporting_records": [],
            }
        },
    )
    assert gate(ctx) is not None

    ctx_bad = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "confidence_level": "insufficient_data",
                "efficacy_score": 0.5,
            }
        },
    )
    with pytest.raises(ValueError, match="must be unset"):
        gate(ctx_bad)


def test_report_validation_raises_on_empty_supporting_records():
    """ReportValidationHook should raise when supporting_records is empty."""
    gate = ReportValidationHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "efficacy_score": 0.5,
                "confidence_score": 0.2,
                "supporting_records": [],
            }
        },
    )
    with pytest.raises(ValueError, match="supporting_records must be a non-empty"):
        gate(ctx)


def test_report_validation_passes_on_valid_report():
    """ReportValidationHook should pass when all fields are valid."""
    gate = ReportValidationHook()
    ctx = HookContext(
        runtime=None,
        task_spec={},
        extra={
            "FinalReport": {
                "efficacy_score": 0.5,
                "confidence_score": 0.2,
                "supporting_records": ["rec-1"],
            }
        },
    )
    result = gate(ctx)
    assert result is not None  # context passed through


def test_report_validation_noop_on_missing_report():
    """ReportValidationHook should no-op when no FinalReport is present."""
    gate = ReportValidationHook()
    ctx = HookContext(runtime=None, task_spec={}, extra={})
    result = gate(ctx)
    assert result is ctx  # identity pass-through for no-op


# ---------------------------------------------------------------------------
# _build_final_report
# ---------------------------------------------------------------------------


def test_build_final_report_has_expected_keys():
    spec = {"drug_ids": ["DB1"], "disease_id": "Cancer", "endpoints": ["survival"]}
    report = _build_final_report({"overall_conclusion": "test"}, spec)
    entry = report["DB1"]["Cancer"]["survival"]
    assert entry["efficacy_score"] == 0.5
    assert entry["confidence_score"] == 0.2
    assert entry["supporting_records"] == ["stub-record-1"]


def test_build_final_report_overrides_from_task_spec():
    """I5: _build_final_report respects override keys in task_spec."""
    spec = {
        "drug_ids": ["DB1"],
        "disease_id": "Cancer",
        "endpoints": ["survival"],
        "_efficacy_score_override": 0.1,
        "_confidence_score_override": 0.9,
        "_supporting_records_override": ["custom-rec"],
    }
    report = _build_final_report({"overall_conclusion": "test"}, spec)
    entry = report["DB1"]["Cancer"]["survival"]
    assert entry["efficacy_score"] == 0.1
    assert entry["confidence_score"] == 0.9
    assert entry["supporting_records"] == ["custom-rec"]
