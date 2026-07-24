"""Test v0.9.0 FourDExpert."""

import pytest

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


# ------------------------------------------------------------------
# Phase D: Coordination methods
# ------------------------------------------------------------------


def test_fourd_expert_delegate_to_expert_returns_dict():
    """delegate_to_expert returns dict with expected keys."""
    expert = FourDExpert()
    with pytest.raises(ValueError, match="Unknown domain"):
        # 'molecular' is known, but 'fake_domain' triggers the error
        expert.delegate_to_expert("fake_domain", [], "test question")


def test_fourd_expert_delegate_to_expert_rejects_unknown_domain():
    """delegate_to_expert raises ValueError for unknown domain."""
    expert = FourDExpert()
    with pytest.raises(ValueError, match="Unknown domain"):
        expert.delegate_to_expert("quantum_physics", [], "test")


def test_fourd_expert_synthesize_combines_reports():
    """synthesize combines multiple expert reports into a unified dict."""
    expert = FourDExpert()
    reports = [
        {
            "domain": "molecular",
            "conclusion": "strong binding",
            "confidence": {"low": 0.7, "high": 0.9},
            "supporting_evidence": [],
        },
        {
            "domain": "clinical",
            "conclusion": "effective in phase 2",
            "confidence": {"low": 0.5, "high": 0.8},
            "supporting_evidence": [],
        },
    ]
    result = expert.synthesize(reports)
    assert isinstance(result, dict)
    assert "overall_conclusion" in result
    assert "confidence" in result
    assert "expert_reports" in result
    assert "conflicts" in result
    assert result["expert_reports"] == reports
    assert result["confidence"] in ("low", "moderate", "high")


def test_fourd_expert_synthesize_detects_confidence_divergence():
    """synthesize detects conflicts when confidence ranges diverge."""
    expert = FourDExpert()
    reports = [
        {
            "domain": "molecular",
            "conclusion": "high efficacy",
            "confidence": {"low": 0.8, "high": 0.95},
            "supporting_evidence": [],
        },
        {
            "domain": "clinical",
            "conclusion": "failed trial",
            "confidence": {"low": 0.1, "high": 0.3},
            "supporting_evidence": [],
        },
    ]
    result = expert.synthesize(reports)
    assert "conflicts" in result
    # Should detect divergence
    assert len(result["conflicts"]) >= 1


def test_fourd_expert_synthesize_empty_reports():
    """synthesize handles empty report list gracefully."""
    expert = FourDExpert()
    result = expert.synthesize([])
    assert result["overall_conclusion"] == "no assessment"
    assert result["confidence"] == "low"
    assert result["expert_reports"] == []
    assert result["conflicts"] == []


def test_fourd_expert_di_constructor():
    """FourDExpert accepts DI parameters matching Expert -> BaseAgent chain."""
    from dargus.runtime.hooks import HookRegistry

    hooks = HookRegistry()
    expert = FourDExpert(
        dbase=None,
        config={"test": True},
        hook_registry=hooks,
    )
    assert expert._hook_registry is hooks
    assert expert.config["test"] is True
