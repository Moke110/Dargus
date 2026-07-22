"""Test v0.15.0 ClinicExpert."""

from dargus.experts.clinic import ClinicExpert
from dargus.experts.protocol import ExpertContext, ExpertReport


def _make_record(**overrides):
    r = {
        "evidence_id": "ev_test",
        "biological_level": "rct",
        "drug_id": "D1",
        "disease_id": "Dis1",
        "readout_value": 0.5,
        "phase": "phase_2",
    }
    r.update(overrides)
    return r


def test_clinic_expert_supported_levels():
    expert = ClinicExpert()
    assert expert.SUPPORTED_LEVELS == ("rct", "epi", "rct-sim")


def test_clinic_expert_can_handle_clinical():
    expert = ClinicExpert()
    record = _make_record(biological_level="rct")
    assert expert.can_handle(record) is True


def test_clinic_expert_can_handle_clinical_sim():
    expert = ClinicExpert()
    record = _make_record(biological_level="rct-sim")
    assert expert.can_handle(record) is True


def test_clinic_expert_rejects_molecular():
    expert = ClinicExpert()
    record = _make_record(biological_level="molecular")
    assert expert.can_handle(record) is False


def test_clinic_expert_assess_returns_report():
    expert = ClinicExpert()
    record = _make_record()
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert isinstance(report, ExpertReport)
    assert report.expert == "ClinicExpert"


def test_clinic_expert_delegates_molecular():
    expert = ClinicExpert()
    record = _make_record(biological_level="molecular")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    deleg_targets = [d.target_expert for d in report.delegations]
    assert "MoleculeExpert" in deleg_targets


def test_clinic_expert_phase_3_weights_higher():
    expert = ClinicExpert()
    record_p2 = _make_record(evidence_id="ev_p2", phase="phase_2")
    record_p3 = _make_record(evidence_id="ev_p3", phase="phase_3")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record_p2, record_p3], ctx)
    scores = {f.record_ids[0]: f.quality_score for f in report.findings}
    assert scores["ev_p3"] > scores["ev_p2"]
