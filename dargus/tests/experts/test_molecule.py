"""Test v0.15.0 MolecularExpert."""

from dargus.experts.molecule import MoleculeExpert
from dargus.experts.protocol import ExpertContext, ExpertReport


def _make_record(**overrides):
    r = {
        "evidence_id": "ev_test",
        "biological_level": "molecular",
        "drug_id": "D1",
        "disease_id": "Dis1",
        "readout_value": 1.0,
    }
    r.update(overrides)
    return r


def test_molecule_expert_supported_levels():
    expert = MoleculeExpert()
    assert "molecular" in expert.SUPPORTED_LEVELS
    assert "molecular-sim" in expert.SUPPORTED_LEVELS
    assert len(expert.SUPPORTED_LEVELS) == 2


def test_molecule_expert_can_handle_molecular():
    expert = MoleculeExpert()
    record = _make_record(biological_level="molecular")
    assert expert.can_handle(record) is True


def test_molecule_expert_can_handle_molecular_sim():
    expert = MoleculeExpert()
    record = _make_record(biological_level="molecular-sim")
    assert expert.can_handle(record) is True


def test_molecule_expert_rejects_clinical():
    expert = MoleculeExpert()
    record = _make_record(biological_level="rct")
    assert expert.can_handle(record) is False


def test_molecule_expert_assess_returns_report():
    expert = MoleculeExpert()
    record = _make_record(biological_level="molecular", readout_value=0.5)
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert isinstance(report, ExpertReport)
    assert report.expert == "MoleculeExpert"
    assert report.round == 1
    assert len(report.findings) >= 1
    assert report.confidence.low >= 0.0
    assert report.confidence.high <= 1.0


def test_molecule_expert_delegates_clinical():
    expert = MoleculeExpert()
    record = _make_record(biological_level="rct")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert len(report.delegations) >= 1
    assert report.delegations[0].target_expert == "ClinicExpert"
