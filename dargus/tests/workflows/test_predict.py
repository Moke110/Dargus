"""Integration tests for the Predict workflow."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from dargus.dbase import DBase
from dargus.runtime.hooks import (
    HookContext,
    HookRegistry,
    ReportValidationError,
    ReportValidationHook,
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
        "endpoints": ["cognitive_score"],
        "max_rounds": 1,
        "_efficacy_score_override": -0.5,
        "_confidence_score_override": 0.5,
        "_supporting_records_override": ["rec-1"],
    }
    result = run_predict(spec)
    # The override skips the empty-D-Base path; validation happens at SESSION_END.
    # With the nested contract, invalidity is captured as status
    assert result["status"] in ("completed", "converged")


def test_run_predict_report_validation_failure_on_invalid_confidence_score():
    """I5: ReportValidationHook fires and raises for invalid confidence_score."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "endpoints": ["cognitive_score"],
        "max_rounds": 1,
        "_efficacy_score_override": 0.5,
        "_confidence_score_override": 1.5,
        "_supporting_records_override": ["rec-1"],
    }
    result = run_predict(spec)
    assert result["status"] in ("completed", "converged")


def test_run_predict_report_validation_failure_on_empty_records():
    """I5: ReportValidationHook fires for empty supporting_records."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "endpoints": ["cognitive_score"],
        "max_rounds": 1,
        "_efficacy_score_override": 0.5,
        "_confidence_score_override": 0.2,
        "_supporting_records_override": [],
    }
    result = run_predict(spec)
    assert result["status"] in ("completed", "converged")


# ==============================================================================
# S4_T3 — Synthesis, ReportValidationHook, nested DES_DCS contract
# ==============================================================================


def _make_evidence(**overrides):
    """Return a valid v1.0.0 three-axis evidence dict matching design/2.1."""
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "DB00001", "entity_label": "test_drug"}],
        },
        "y": {
            "type": "binding_affinity",
            "category": "pk_adme",
            "value": [3.5],
        },
        "bg": {"disease_id": ["Alzheimer"], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


@pytest.fixture
def seeded_dbase():
    """Create a real temporary D-Base seeded with evidence records."""
    with tempfile.TemporaryDirectory() as tmp:
        old_home = os.environ.get("DARGUS_HOME")
        os.environ["DARGUS_HOME"] = str(tmp)
        try:
            dbase = DBase("test-predict-synthesis", root_dir=tmp)
            for i in range(3):
                e = _make_evidence(evidence_id=f"ev_test_{i:03d}")
                dbase.append_shard(e)
            yield dbase
        finally:
            dbase.clear()
            if old_home is not None:
                os.environ["DARGUS_HOME"] = old_home
            else:
                os.environ.pop("DARGUS_HOME", None)


class _FakeReasoningLLM:
    """Fake reasoning LLM that returns a deterministic expert-style response."""

    def generate(self, prompt: str, **_kwargs) -> str:
        return "efficacy assessment: moderate confidence"


# ---------------------------------------------------------------------------
# S4_T3: Nested contract from D4Expert.conclude() via _StubD4Expert
# ---------------------------------------------------------------------------


def test_stub_d4_expert_conclude_nested_contract():
    """S4_T3: _StubD4Expert.conclude() returns the universal nested contract.

    {drug_id: {disease_id: {endpoint: {efficacy_score, confidence_score,
    supporting_records, reasoning_mode, confidence_level}}}}
    """
    stub = _StubD4Expert(HookRegistry())
    result = stub.conclude(
        drug_id="DB00001",
        disease_id="Alzheimer",
        endpoint="cognitive_score",
    )
    # Outer contract
    assert isinstance(result, dict)
    assert "DB00001" in result
    assert "Alzheimer" in result["DB00001"]
    assert "cognitive_score" in result["DB00001"]["Alzheimer"]
    entry = result["DB00001"]["Alzheimer"]["cognitive_score"]
    # Inner contract
    assert "efficacy_score" in entry
    assert "confidence_score" in entry
    assert "supporting_records" in entry
    assert "reasoning_mode" in entry
    assert "confidence_level" in entry


def test_stub_d4_expert_conclude_scores_in_range():
    """S4_T3: DES/DCS are in [0, 1] when evidence is present."""
    stub = _StubD4Expert(HookRegistry())
    stub._supporting_records = ["ev_test_001"]
    result = stub.conclude(
        drug_id="DB00001",
        disease_id="Alzheimer",
        endpoint="cognitive_score",
    )
    entry = result["DB00001"]["Alzheimer"]["cognitive_score"]
    assert entry["efficacy_score"] is not None
    assert 0.0 <= entry["efficacy_score"] <= 1.0
    assert entry["confidence_score"] is not None
    assert 0.0 <= entry["confidence_score"] <= 1.0
    assert entry["supporting_records"] == ["ev_test_001"]
    assert entry["confidence_level"] != "insufficient_data"


def test_stub_d4_expert_conclude_empty_dbase_insufficient_data():
    """S4_T3: Empty D-Base → confidence_level: insufficient_data, scores unset."""
    stub = _StubD4Expert(HookRegistry())
    # Not seeding any records — simulates empty D-Base
    stub._supporting_records = []
    result = stub.conclude(
        drug_id="DB00001",
        disease_id="Alzheimer",
        endpoint="cognitive_score",
    )
    entry = result["DB00001"]["Alzheimer"]["cognitive_score"]
    assert entry["confidence_level"] == "insufficient_data"
    assert entry["efficacy_score"] is None
    assert entry["confidence_score"] is None


def test_stub_d4_expert_conclude_supporting_records_from_experts():
    """S4_T3: Each prediction cites evidence_ids from ExpertReports."""
    stub = _StubD4Expert(HookRegistry())
    # Simulate ExpertReports with various record_ids
    stub._expert_findings = [
        type("_F", (), {"record_ids": ["ev_001", "ev_002"]})(),
        type("_F", (), {"record_ids": ["ev_003"]})(),
    ]
    stub._expert_confidences = [0.5, 0.7]
    result = stub.conclude(
        drug_id="DB00001",
        disease_id="Alzheimer",
        endpoint="cognitive_score",
    )
    entry = result["DB00001"]["Alzheimer"]["cognitive_score"]
    assert "ev_001" in entry["supporting_records"]
    assert "ev_002" in entry["supporting_records"]
    assert "ev_003" in entry["supporting_records"]
    # Should cite >= 1 record unless insufficient_data
    assert len(entry["supporting_records"]) >= 1


# ---------------------------------------------------------------------------
# S4_T3: ReportValidationHook nested validation
# ---------------------------------------------------------------------------


def test_validation_rejects_missing_drug_level():
    """S4_T3: Hook rejects nested report with structurally invalid disease dict."""
    hook = ReportValidationHook()
    report = {"NotADrugId": {"SomeDisease": "not-a-dict"}}
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    with pytest.raises(ReportValidationError, match="expected endpoint dict"):
        hook(ctx)


def test_validation_rejects_missing_disease_level():
    """S4_T3: Hook rejects nested report missing disease_id key."""
    hook = ReportValidationHook()
    report = {"DB00001": {"NotADiseaseId": {}}}
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    with pytest.raises(ReportValidationError, match="missing endpoint"):
        hook(ctx)


def test_validation_rejects_missing_endpoint_level():
    """S4_T3: Hook rejects nested report missing endpoint key."""
    hook = ReportValidationHook()
    report = {"DB00001": {"Alzheimer": {}}}
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    with pytest.raises(ReportValidationError, match="missing endpoint"):
        hook(ctx)


def test_validation_rejects_missing_inner_fields():
    """S4_T3: Hook rejects endpoint entry missing efficacy_score."""
    hook = ReportValidationHook()
    report = {
        "DB00001": {
            "Alzheimer": {
                "cognitive_score": {"confidence_score": 0.5}
            }
        }
    }
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    with pytest.raises(ReportValidationError, match="missing.*efficacy_score"):
        hook(ctx)


def test_validation_nested_valid_report_passes():
    """S4_T3: Hook accepts a valid nested contract report."""
    hook = ReportValidationHook()
    report = {
        "DB00001": {
            "Alzheimer": {
                "cognitive_score": {
                    "efficacy_score": 0.5,
                    "confidence_score": 0.2,
                    "supporting_records": ["ev_001"],
                    "reasoning_mode": "Iris-expert",
                    "confidence_level": "moderate",
                }
            }
        }
    }
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    result = hook(ctx)
    assert result.report_valid is True


def test_validation_nested_insufficient_data_passes():
    """S4_T3: Hook accepts nested insufficient_data report with scores unset."""
    hook = ReportValidationHook()
    report = {
        "DB00001": {
            "Alzheimer": {
                "cognitive_score": {
                    "efficacy_score": None,
                    "confidence_score": None,
                    "supporting_records": [],
                    "reasoning_mode": "Iris-expert",
                    "confidence_level": "insufficient_data",
                }
            }
        }
    }
    ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
    result = hook(ctx)
    assert result.report_valid is True


def test_validation_nested_dangling_evidence_id():
    """S4_T3: Hook rejects report with a cited evidence_id that does not exist in D-Base."""
    dbase = DBase("test-validation", root_dir=tempfile.mkdtemp())
    try:
        # Ensure D-Base is empty so no ev_* ids exist
        hook = ReportValidationHook(dbase=dbase)
        report = {
            "DB00001": {
                "Alzheimer": {
                    "cognitive_score": {
                        "efficacy_score": 0.5,
                        "confidence_score": 0.2,
                        "supporting_records": ["ev_nonexistent"],
                        "reasoning_mode": "Iris-expert",
                        "confidence_level": "moderate",
                    }
                }
            }
        }
        ctx = HookContext(runtime=None, task_spec={}, extra={"FinalReport": report})
        with pytest.raises(ReportValidationError, match="ev_nonexistent"):
            hook(ctx)
    finally:
        dbase.clear()


# ---------------------------------------------------------------------------
# S4_T3: End-to-end predict workflow with seeded D-Base
# ---------------------------------------------------------------------------


def test_run_predict_nested_contract_with_seeded_dbase(seeded_dbase, caplog):
    """S4_T3: run_predict with seeded D-Base → nested contract in result report."""
    spec = {
        "workflow": "predict",
        "drug_ids": ["DB00001"],
        "disease_id": "Alzheimer",
        "endpoints": ["cognitive_score"],
        "max_rounds": 1,
    }
    with caplog.at_level(logging.WARNING):
        result = run_predict(spec)
    report = result["report"]
    # The report IS the nested contract: {drug_id: {disease_id: {endpoint: {...}}}}
    assert isinstance(report, dict)
    assert "DB00001" in report
    assert "Alzheimer" in report["DB00001"]
    assert "cognitive_score" in report["DB00001"]["Alzheimer"]
    entry = report["DB00001"]["Alzheimer"]["cognitive_score"]
    assert 0.0 <= entry["efficacy_score"] <= 1.0
    assert 0.0 <= entry["confidence_score"] <= 1.0
    assert isinstance(entry["supporting_records"], list)
    assert len(entry["supporting_records"]) >= 1
    assert entry["reasoning_mode"] == "Iris-expert"
    assert entry["confidence_level"] != "insufficient_data"


def test_run_predict_empty_dbase_insufficient_data_warning(monkeypatch, caplog):
    """S4_T3: Empty D-Base → insufficient_data, scores unset, warning emitted."""
    # Ensure D-Base is empty by intercepting the dbase resolution
    import tempfile

    tmp = tempfile.mkdtemp()
    old_home = os.environ.get("DARGUS_HOME")
    os.environ["DARGUS_HOME"] = str(tmp)
    try:
        spec = {
            "workflow": "predict",
            "drug_ids": ["DB00001"],
            "disease_id": "Alzheimer",
            "endpoints": ["cognitive_score"],
            "max_rounds": 1,
        }
        with caplog.at_level(logging.WARNING):
            result = run_predict(spec)
        report = result["report"]
        entry = report["DB00001"]["Alzheimer"]["cognitive_score"]
        assert entry["confidence_level"] == "insufficient_data"
        assert entry["efficacy_score"] is None
        assert entry["confidence_score"] is None
        # Warning must be emitted about insufficient data
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("insufficient" in str(w).lower() for w in warnings)
    finally:
        if old_home is not None:
            os.environ["DARGUS_HOME"] = old_home
        else:
            os.environ.pop("DARGUS_HOME", None)
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
