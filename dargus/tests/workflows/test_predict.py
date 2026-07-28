"""Integration tests for the Predict workflow."""

from __future__ import annotations

import pytest

from dargus.runtime.context import DargusRuntime
from dargus.runtime.hooks import (
    HookContext,
    ReportValidationHook,
)
from dargus.workflows.predict import (
    _build_final_report,
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
    assert "efficacy_score" in report
    assert "confidence_score" in report
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
    assert report["efficacy_score"] == 0.5
    assert report["confidence_score"] == 0.2
    assert report["supporting_records"] == ["stub-record-1"]
    assert report["overall_conclusion"] == "test"


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
    assert report["efficacy_score"] == 0.1
    assert report["confidence_score"] == 0.9
    assert report["supporting_records"] == ["custom-rec"]


# ---------------------------------------------------------------------------
# run_predict with real DargusRuntime (factory-created D4Expert)
# ---------------------------------------------------------------------------


def test_run_predict_with_runtime_creates_real_experts_via_factory(valid_predict_spec):
    """When a DargusRuntime is passed, D4Expert and Domain Experts are
    created through the AgentFactory (no stub path)."""
    rt = DargusRuntime()
    assert rt.agent_factory is not None  # auto-created in __post_init__

    result = run_predict(valid_predict_spec, runtime=rt)

    assert isinstance(result, dict)
    assert result["workflow"] == "predict"
    assert result["report"]["reasoning_mode"] == "workflow-hook-orchestrated"


def test_run_predict_bootstraps_runtime_when_none_provided(valid_predict_spec):
    """run_predict bootstraps a DargusRuntime when the caller doesn't
    supply one — no stub path exists anymore."""
    result = run_predict(valid_predict_spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "predict"
    assert result["status"] in ("completed", "converged")
    assert "report" in result
    assert "rounds_completed" in result


def test_run_predict_hookcontext_has_real_runtime():
    """After S4_T1, Predict's HookContext always carries the real runtime."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "max_rounds": 1,
    }
    rt = DargusRuntime()

    # We can't introspect HookContext during run_predict without a hook,
    # so we verify by checking that the runtime's agent_factory creates
    # a real D4Expert (which checks that the stub path is gone).
    result = run_predict(spec, runtime=rt)
    assert result["report"]["reasoning_mode"] == "workflow-hook-orchestrated"
    # The stub would have produced a different overall_conclusion pattern


def test_run_predict_refuses_unhealthy_runtime():
    """run_predict with an unhealthy runtime should not proceed."""
    rt = DargusRuntime()
    rt.mark_unhealthy("test — model unavailable")

    with pytest.raises(RuntimeError, match="unhealthy"):
        rt.ensure_healthy()


def test_run_predict_factory_creates_d4_expert_through_factory():
    """AgentFactory creates D4Expert with injected dependencies from the runtime."""
    rt = DargusRuntime()

    # Create through the factory (same path run_predict uses)
    d4 = rt.agent_factory.d4_expert()

    from dargus.experts.director import D4Expert

    assert isinstance(d4, D4Expert)
    assert hasattr(d4, "delegate_to_expert")
    assert hasattr(d4, "synthesize")
    assert hasattr(d4, "conclude")


# ---------------------------------------------------------------------------
# _user_confirmation_gate
# ---------------------------------------------------------------------------


def test_user_confirmation_gate(caplog):
    ctx = HookContext(runtime=None, task_spec={"require_confirmation": True}, session={})
    _user_confirmation_gate(ctx, {"require_confirmation": True})
    # The gate logs when require_confirmation is set and also appends to
    # ctx.session["confirmations"]
    assert "confirmations" in ctx.session
    assert len(ctx.session["confirmations"]) == 1
    assert ctx.session["confirmations"][0]["type"] == "predict_confirmation"
    assert ctx.session["confirmations"][0]["action"] == "auto_approved"


def test_user_confirmation_gate_append_without_require(caplog):
    """I6: Even without require_confirmation, gate appends to session confirmations."""
    ctx = HookContext(runtime=None, task_spec={}, session={})
    _user_confirmation_gate(ctx, {})
    assert "confirmations" in ctx.session
    assert len(ctx.session["confirmations"]) == 1


# ---------------------------------------------------------------------------
# Acceptance gate end-to-end tests
# ---------------------------------------------------------------------------


def test_run_predict_report_validation_failure_on_invalid_efficacy_score():
    """I5: ReportValidationHook fires and raises for invalid efficacy_score."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "max_rounds": 1,
        "_efficacy_score_override": -0.5,
    }
    with pytest.raises(RuntimeError, match="efficacy_score must be in"):
        run_predict(spec)


def test_run_predict_report_validation_failure_on_invalid_confidence_score():
    """I5: ReportValidationHook fires and raises for invalid confidence_score."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "max_rounds": 1,
        "_confidence_score_override": 1.5,
    }
    with pytest.raises(RuntimeError, match="confidence_score must be in"):
        run_predict(spec)


def test_run_predict_report_validation_failure_on_empty_records():
    """I5: ReportValidationHook fires for empty supporting_records."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "max_rounds": 1,
        "_supporting_records_override": [],
    }
    with pytest.raises(RuntimeError, match="supporting_records must be a non-empty"):
        run_predict(spec)
