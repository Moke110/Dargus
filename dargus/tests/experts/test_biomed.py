"""Test v0.15.0 BiomedExpert."""

from dargus.experts.biomed import BiomedExpert
from dargus.experts.protocol import ExpertContext, ExpertReport


def _make_record(**overrides):
    r = {
        "evidence_id": "ev_test",
        "biological_level": "cellular",
        "drug_id": "D1",
        "disease_id": "Dis1",
        "readout_value": 0.5,
    }
    r.update(overrides)
    return r


def test_biomed_expert_supported_levels():
    expert = BiomedExpert()
    expected = ("cellular", "cellular-sim", "exvivo", "exvivo-sim", "animal", "animal-sim")
    assert expert.SUPPORTED_LEVELS == expected
    assert len(expert.SUPPORTED_LEVELS) == 6


def test_biomed_expert_can_handle_cellular():
    expert = BiomedExpert()
    record = _make_record(biological_level="cellular")
    assert expert.can_handle(record) is True


def test_biomed_expert_can_handle_animal_sim():
    expert = BiomedExpert()
    record = _make_record(biological_level="animal-sim")
    assert expert.can_handle(record) is True


def test_biomed_expert_rejects_molecular():
    expert = BiomedExpert()
    record = _make_record(biological_level="molecular")
    assert expert.can_handle(record) is False


def test_biomed_expert_assess_returns_report():
    expert = BiomedExpert()
    record = _make_record(biological_level="cellular")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert isinstance(report, ExpertReport)
    assert report.expert == "BiomedExpert"
    assert len(report.findings) >= 1


def test_biomed_expert_delegates_molecular():
    expert = BiomedExpert()
    record = _make_record(biological_level="molecular")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    deleg_targets = [d.target_expert for d in report.delegations]
    assert "MoleculeExpert" in deleg_targets


def test_biomed_expert_flags_sim_data():
    expert = BiomedExpert()
    record = _make_record(biological_level="cellular-sim")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    sim_bias = [n for n in report.bias_notes if "simulation" in n.lower()]
    assert len(sim_bias) >= 1
