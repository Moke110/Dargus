"""Consolidation tests for the shared Expert assess() template method (#98, #99).

Locks in the behaviour-preservation contract: the four domain Experts now share
one loop spine on the Expert base; their differences are declared as class
attributes (SIM_PENALTY / SIM_BIAS_MSG / RELEVANCE_MAP) and two hooks
(_gate, _collect_gaps). These tests assert the *observable* report deltas, not
the private hook implementations.
"""

from dargus.experts.bioinfo import BioinfoExpert
from dargus.experts.biomed import BiomedExpert
from dargus.experts.clinic import ClinicExpert
from dargus.experts.molecule import MoleculeExpert
from dargus.experts.protocol import ExpertContext


def _ctx():
    return ExpertContext(drug_ids=["D1"], disease_id="Dis1", endpoints=["EP1"], round=1)


def _record(**overrides):
    r = {
        "evidence_id": "ev_test",
        "biological_level": "cellular",
        "readout_value": 0.5,
    }
    r.update(overrides)
    return r


# ------------------------------------------------------------------
# Declarative attributes produce the same report deltas as the old inline code
# ------------------------------------------------------------------


def test_molecule_and_biomed_share_default_sim_penalty():
    """Molecule/Biomed use the base 0.2 sim penalty; Clinic uses 0.3."""
    assert MoleculeExpert.SIM_PENALTY == 0.2
    assert BiomedExpert.SIM_PENALTY == 0.2
    assert ClinicExpert.SIM_PENALTY == 0.3


def test_clinic_sim_penalty_is_applied():
    """Clinic sim records are penalised by 0.3 (vs 0.2 elsewhere)."""
    expert = ClinicExpert()
    sim_record = _record(biological_level="rct-sim", phase="phase_2", readout_value=0.5)
    real_record = _record(biological_level="rct", phase="phase_2", readout_value=0.5)
    sim_report = expert.assess([sim_record], _ctx())
    real_report = expert.assess([real_record], _ctx())
    sim_finding = sim_report.findings[0]
    real_finding = real_report.findings[0]
    # Same base quality (phase_2 + readout); the sim record is penalised by
    # Clinic's 0.3 while the real record keeps full quality.
    assert real_finding.quality_score == sim_finding.quality_score + 0.3
    assert any("simulation" in n for n in sim_report.bias_notes)
    assert real_report.bias_notes == []


def test_clinic_rct_epi_relevance_is_high():
    """Clinic maps rct/epi relevance to high; sim and others medium."""
    expert = ClinicExpert()
    records = [
        _record(biological_level="rct", evidence_id="ev_rct"),
        _record(biological_level="epi", evidence_id="ev_epi"),
        _record(biological_level="rct-sim", evidence_id="ev_sim"),
    ]
    report = expert.assess(records, _ctx())
    relevance = {f.record_ids[0]: f.relevance for f in report.findings}
    assert relevance["ev_rct"] == "high"
    assert relevance["ev_epi"] == "high"
    assert relevance["ev_sim"] == "medium"


def test_biomed_sim_records_carry_bias_note():
    """Biomed's default sim bias note is applied to sim records."""
    expert = BiomedExpert()
    record = _record(biological_level="cellular-sim")
    report = expert.assess([record], _ctx())
    assert any("simulation" in n.lower() for n in report.bias_notes)


# ------------------------------------------------------------------
# _collect_gaps hook outputs (#99)
# ------------------------------------------------------------------


def test_biomed_gap_hook_flags_missing_in_vivo():
    """Biomed: only in-vitro evidence -> 'no in vivo' data gap."""
    expert = BiomedExpert()
    record = _record(biological_level="cellular")
    report = expert.assess([record], _ctx())
    assert any("No in vivo" in g for g in report.data_gaps)


def test_biomed_no_gap_when_in_vivo_present():
    """Biomed: animal evidence present -> no in-vivo gap."""
    expert = BiomedExpert()
    records = [
        _record(biological_level="animal", evidence_id="ev_animal"),
        _record(biological_level="cellular", evidence_id="ev_cell"),
    ]
    report = expert.assess(records, _ctx())
    assert not any("No in vivo" in g for g in report.data_gaps)


def test_molecule_gap_hook_is_noop():
    """Molecule inherits the no-op _collect_gaps: no gaps from pure molecular
    evidence."""
    expert = MoleculeExpert()
    record = _record(biological_level="molecular")
    report = expert.assess([record], _ctx())
    assert report.data_gaps == []
    assert report.bias_notes == []


def test_clinic_gap_hook_flags_no_real_clinical_evidence():
    """Clinic: only sim clinical evidence -> 'no real clinical trial evidence'
    gap."""
    expert = ClinicExpert()
    record = _record(biological_level="rct-sim")
    report = expert.assess([record], _ctx())
    assert any("No real clinical trial evidence" in g for g in report.data_gaps)


def test_clinic_gap_hook_mixed_direction_detection():
    """Clinic: readouts straddling zero -> mixed-direction bias note."""
    expert = ClinicExpert()
    records = [
        _record(biological_level="rct", evidence_id="ev_pos", readout_value=0.5),
        _record(biological_level="rct", evidence_id="ev_neg", readout_value=-0.3),
    ]
    report = expert.assess(records, _ctx())
    assert any("Mixed clinical effect directions" in n for n in report.bias_notes)


# ------------------------------------------------------------------
# Bioinfo _gate hook (#99)
# ------------------------------------------------------------------


def test_bioinfo_gate_routes_non_high_throughput_to_delegation():
    """Bioinfo's _gate admits only high-throughput records; anything else is
    delegated to the level's Expert, exactly as the old inline check did."""
    expert = BioinfoExpert()
    record = _record(biological_level="cellular", assay_type="cell_viability")
    report = expert.assess([record], _ctx())
    assert report.findings == []
    assert len(report.delegations) == 1
    assert report.delegations[0].target_expert == "BiomedExpert"


def test_bioinfo_gate_keeps_high_throughput_sim_record():
    """Bioinfo assesses high-throughput sim records without a sim penalty."""
    expert = BioinfoExpert()
    record = _record(biological_level="molecular-sim", assay_type="rna_seq")
    report = expert.assess([record], _ctx())
    assert len(report.findings) == 1
    assert report.bias_notes == []


def test_bioinfo_gate_delegates_with_distinct_reason():
    """Bioinfo delegation reason names the non-high-throughput cause."""
    expert = BioinfoExpert()
    record = _record(biological_level="cellular", assay_type="cell_viability")
    report = expert.assess([record], _ctx())
    assert report.delegations[0].reason.startswith("Non-high-throughput data delegated to")
