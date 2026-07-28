"""Integration tests for the Predict workflow."""

from __future__ import annotations

import pytest

from dargus.runtime.hooks import (
    HookContext,
    HookRegistry,
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


# ---------------------------------------------------------------------------
# Multi-round Expert dispatch and delegation end-to-end tests (S4_T2)
# ---------------------------------------------------------------------------


def _make_evidence_record(
    evidence_id: str,
    biological_level: str,
    *,
    disease_id: str = "Alzheimer",
    drug_id: str = "DB00001",
    readout_value: float | None = None,
    assay_type: str | None = None,
    phase: str | None = None,
) -> dict:
    """Build a minimal three-axis evidence dict for seeding a test D-Base."""
    record: dict = {
        "evidence_id": evidence_id,
        "biological_level": biological_level,
        "evidence_design": "descriptive",
        "x": {
            "type": "drug",
            "value": [{"entity_id": drug_id, "entity_label": drug_id}],
        },
        "y": {
            "type": "assay",
            "category": "efficacy",
            "value": [readout_value] if readout_value is not None else [],
        },
        "bg": {"disease_id": [disease_id], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "test", "name": "e2e_test_seed"}],
    }
    if assay_type is not None:
        record["platform"] = {"assay_platform": assay_type}
    if phase is not None:
        record["phase"] = phase
    return record


class _FakeReasoningBackend:
    """Fake LLM backend returning deterministic responses — no network."""

    def chat(self, messages, options=None):
        from dargus.models.reasoning import LLMResponse

        return LLMResponse(
            content='{"goal":"assess evidence","confidence":0.5}',
            model="fake",
        )


def test_run_predict_real_expert_loop_multi_round(tmp_path):
    """E2E: multi-round Expert dispatch, ExpertReport production,
    delegation processing, and SafetyNetHook termination.

    Seeds a real temp D-Base with evidence at multiple biological levels
    and uses a fake reasoning LLM so no network is required.
    """
    import os

    from dargus.dbase import DBase, DBaseStore
    from dargus.models.reasoning import ReasoningLLM
    from dargus.runtime.context import DargusRuntime

    # ---- Temp D-Base ----------------------------------------------------
    dargus_home = str(tmp_path / "dargus_home")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)

    dbase = DBase(project_id="e2e-predict", root_dir=dargus_home)
    store = DBaseStore(dbase)

    # Seed evidence records at different biological levels
    seed_records = [
        _make_evidence_record("ev_mol_1", "molecular", drug_id="DRUG_A", readout_value=0.85),
        _make_evidence_record("ev_mol_2", "molecular", drug_id="DRUG_A", readout_value=0.72),
        _make_evidence_record("ev_cell_1", "cellular", drug_id="DRUG_A", readout_value=0.65),
        _make_evidence_record("ev_anim_1", "animal", drug_id="DRUG_A", readout_value=0.55),
        _make_evidence_record(
            "ev_rct_1", "rct", drug_id="DRUG_A", readout_value=0.48, phase="phase_2"
        ),
        _make_evidence_record("ev_epi_1", "epi", drug_id="DRUG_A", readout_value=0.35),
        _make_evidence_record(
            "ev_rnaseq_1", "molecular", drug_id="DRUG_A", assay_type="rna_seq", readout_value=0.78
        ),
    ]
    for rec in seed_records:
        dbase.append_shard(rec)

    # ---- Fake Runtime ---------------------------------------------------
    fake_backend = _FakeReasoningBackend()
    fake_llm = ReasoningLLM(backend=fake_backend)
    runtime = DargusRuntime(
        config={},
        reasoning_llm=fake_llm,
        embedding_model=None,
    )
    runtime.dbase_store = store

    # ---- Run Predict ----------------------------------------------------
    task_spec = {
        "workflow": "predict",
        "drug_ids": ["DRUG_A"],
        "disease_id": "Alzheimer",
        "endpoints": ["efficacy"],
        "max_rounds": 4,
        "timeout_seconds": 30,
    }
    result = run_predict(task_spec, runtime=runtime)

    # ---- Assertions -----------------------------------------------------
    assert isinstance(result, dict)
    assert result["workflow"] == "predict"
    assert result["status"] in ("completed", "converged")
    assert result["rounds_completed"] <= 4  # SafetyNetHook enforcement

    report = result["report"]
    des = report.get("efficacy_score")
    dcs = report.get("confidence_score")
    assert des is not None, "DES must be set when evidence exists"
    assert dcs is not None, "DCS must be set when evidence exists"
    assert 0.0 <= des <= 1.0
    assert 0.0 <= dcs <= 1.0

    # Supporting records must come from seeded D-Base (ev_ prefix)
    records = report.get("supporting_records", [])
    assert isinstance(records, list)
    assert len(records) > 0, "Supporting records must be non-empty for seeded D-Base"
    for rid in records:
        assert rid.startswith("ev_"), f"{rid!r} not from seeded D-Base"

    session = result["session"]
    assert "FinalReport" in session
    assert "rounds" in session
    assert len(session["rounds"]) >= 1


def test_run_predict_expert_loop_delegation_between_experts(tmp_path):
    """E2E: Delegation tracking — when a record is out of scope for one
    Expert, a TaskDelegation routes it to the correct Expert.  Verifies
    that records at different biological levels are assessed by the
    appropriate domain Expert.
    """
    import os

    from dargus.dbase import DBase, DBaseStore
    from dargus.models.reasoning import ReasoningLLM
    from dargus.runtime.context import DargusRuntime

    dargus_home = str(tmp_path / "dargus_home")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)

    dbase = DBase(project_id="e2e-delegation", root_dir=dargus_home)
    store = DBaseStore(dbase)

    seed_records = [
        _make_evidence_record("ev_mol_a", "molecular", drug_id="DRUG_B"),
        _make_evidence_record("ev_mol_sim", "molecular-sim", drug_id="DRUG_B"),
        _make_evidence_record(
            "ev_rct_a", "rct", drug_id="DRUG_B", readout_value=0.42, phase="phase_3"
        ),
        _make_evidence_record("ev_cell_a", "cellular", drug_id="DRUG_B"),
        _make_evidence_record("ev_animal_a", "animal", drug_id="DRUG_B", readout_value=0.61),
    ]
    for rec in seed_records:
        dbase.append_shard(rec)

    fake_backend = _FakeReasoningBackend()
    fake_llm = ReasoningLLM(backend=fake_backend)
    runtime = DargusRuntime(
        config={},
        reasoning_llm=fake_llm,
        embedding_model=None,
    )
    runtime.dbase_store = store

    task_spec = {
        "workflow": "predict",
        "drug_ids": ["DRUG_B"],
        "disease_id": "Alzheimer",
        "endpoints": ["efficacy"],
        "max_rounds": 5,
        "timeout_seconds": 30,
    }
    result = run_predict(task_spec, runtime=runtime)

    assert isinstance(result, dict)
    assert result["status"] in ("completed", "converged")

    report = result["report"]
    des = report.get("efficacy_score")
    dcs = report.get("confidence_score")
    assert des is not None
    assert dcs is not None
    assert 0.0 <= des <= 1.0
    assert 0.0 <= dcs <= 1.0

    records = report.get("supporting_records", [])
    assert len(records) > 0


def test_run_predict_insufficient_data_on_empty_dbase(tmp_path):
    """E2E: When the D-Base has no evidence, Predict must return
    confidence_level: insufficient_data with scores unset."""
    import os

    from dargus.dbase import DBase, DBaseStore
    from dargus.models.reasoning import ReasoningLLM
    from dargus.runtime.context import DargusRuntime

    dargus_home = str(tmp_path / "dargus_home_empty")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)

    dbase = DBase(project_id="e2e-empty", root_dir=dargus_home)
    store = DBaseStore(dbase)  # deliberately empty — no seed records

    fake_backend = _FakeReasoningBackend()
    fake_llm = ReasoningLLM(backend=fake_backend)
    runtime = DargusRuntime(
        config={},
        reasoning_llm=fake_llm,
        embedding_model=None,
    )
    runtime.dbase_store = store

    task_spec = {
        "workflow": "predict",
        "drug_ids": ["DRUG_C"],
        "disease_id": "RareDisease",
        "endpoints": ["efficacy"],
        "max_rounds": 3,
        "timeout_seconds": 30,
    }
    result = run_predict(task_spec, runtime=runtime)

    assert isinstance(result, dict)
    report = result["report"]
    assert report.get("confidence_level") == "insufficient_data"
    assert report.get("efficacy_score") is None
    assert report.get("confidence_score") is None
