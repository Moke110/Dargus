"""Test BioinfoExpert."""

from dargus.experts.bioinfo import BioinfoExpert
from dargus.experts.protocol import ExpertContext, ExpertReport


def _make_record(**overrides):
    r = {
        "evidence_id": "ev_test",
        "biological_level": "molecular",
        "drug_id": "D1",
        "disease_id": "Dis1",
        "readout_value": 0.5,
        "assay_type": "rna_seq",
    }
    r.update(overrides)
    return r


def test_bioinfo_expert_supported_levels():
    expert = BioinfoExpert()
    assert len(expert.SUPPORTED_LEVELS) == 11
    assert "molecular" in expert.SUPPORTED_LEVELS
    assert "rct-sim" in expert.SUPPORTED_LEVELS


def test_bioinfo_expert_can_handle_high_throughput():
    expert = BioinfoExpert()
    record = _make_record(biological_level="molecular", assay_type="rna_seq")
    assert expert._is_high_throughput(record) is True


def test_bioinfo_expert_rejects_low_throughput():
    expert = BioinfoExpert()
    record = _make_record(biological_level="molecular", assay_type="binding_affinity")
    assert expert._is_high_throughput(record) is False


def test_bioinfo_expert_assess_returns_report():
    expert = BioinfoExpert()
    record = _make_record(biological_level="molecular", assay_type="rna_seq")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert isinstance(report, ExpertReport)
    assert report.expert == "BioinfoExpert"


def test_bioinfo_expert_delegates_non_omics_to_level_expert():
    expert = BioinfoExpert()
    record = _make_record(biological_level="cellular", assay_type="cell_viability")
    ctx = ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)
    report = expert.assess([record], ctx)
    assert len(report.delegations) >= 1
