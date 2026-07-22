"""Test v0.9.0 FourDExpert."""

from dargus.experts.director import FourDExpert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertReport,
    FinalReport,
)


def _make_report(expert_name, round_num, quality, level, delegations=None):
    return ExpertReport(
        expert=expert_name,
        round=round_num,
        findings=[
            EvidenceAssessment(
                record_ids=["rec_001"],
                biological_level=level,
                quality_score=quality,
            )
        ],
        confidence=ConfidenceInterval(low=0.3, high=0.7, sources=[]),
        delegations=delegations or [],
        data_gaps=[],
        bias_notes=[],
    )


def test_fourd_expert_has_all_levels():
    expert = FourDExpert()
    assert len(expert.SUPPORTED_LEVELS) == 11


def test_fourd_expert_conclude_returns_final_report():
    expert = FourDExpert()
    reports = [
        _make_report("MoleculeExpert", 1, 0.7, "molecular"),
        _make_report("ClinicExpert", 1, 0.5, "clinical"),
    ]
    result = expert.conclude(
        drug_id="DRUG_A",
        disease_id="DISEASE_X",
        endpoint="primary_endpoint_change",
        all_reports={
            "MoleculeExpert": reports[:1],
            "ClinicExpert": reports[1:],
        },
    )
    assert isinstance(result, FinalReport)
    assert result.drug_id == "DRUG_A"
    assert result.disease_id == "DISEASE_X"


def test_fourd_expert_conclude_synthesizes_contradictions():
    expert = FourDExpert()
    pos_report = ExpertReport(
        expert="BiomedExpert",
        round=1,
        findings=[
            EvidenceAssessment(
                record_ids=["rec_pos"],
                biological_level="animal",
                quality_score=0.8,
            )
        ],
        confidence=ConfidenceInterval(low=0.6, high=0.9, sources=[]),
        data_gaps=[],
        bias_notes=["Positive efficacy in animal models"],
    )
    neg_report = ExpertReport(
        expert="ClinicExpert",
        round=1,
        findings=[
            EvidenceAssessment(
                record_ids=["rec_neg"],
                biological_level="clinical",
                quality_score=0.7,
            )
        ],
        confidence=ConfidenceInterval(low=0.1, high=0.3, sources=[]),
        data_gaps=["Small sample size"],
        bias_notes=["Failed to meet primary endpoint in phase 2"],
    )
    result = expert.conclude(
        drug_id="DRUG_B",
        disease_id="DISEASE_Y",
        endpoint="primary_endpoint_change",
        all_reports={
            "BiomedExpert": [pos_report],
            "ClinicExpert": [neg_report],
        },
    )
    assert len(result.contradictions) >= 1


def test_fourd_expert_generates_guidance():
    expert = FourDExpert()
    reports = [
        _make_report("MoleculeExpert", 1, 0.3, "molecular-sim"),
        _make_report("BiomedExpert", 1, 0.2, "cellular"),
    ]
    guidance = expert.generate_guidance(reports, round_num=1)
    assert isinstance(guidance, str)
    assert len(guidance) > 0
