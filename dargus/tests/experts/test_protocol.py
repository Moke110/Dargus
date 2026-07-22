"""Test v0.9.0 Expert protocol types."""

from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    FinalReport,
    TaskDelegation,
)


def test_evidence_assessment_defaults():
    ea = EvidenceAssessment(
        record_ids=["rec_001"],
        biological_level="molecular",
    )
    assert ea.relevance == "medium"
    assert ea.quality_score == 0.5
    assert ea.limitations == []


def test_task_delegation():
    td = TaskDelegation(
        target_expert="BioinfoExpert",
        record_ids=["rec_002"],
        reason="High-throughput GWAS data detected",
        priority="high",
    )
    assert td.target_expert == "BioinfoExpert"
    assert td.priority == "high"


def test_confidence_interval():
    ci = ConfidenceInterval(low=0.3, high=0.8, sources=["small_sample_size"])
    assert ci.low == 0.3
    assert ci.high == 0.8
    assert "small_sample_size" in ci.sources


def test_expert_report_round_tracking():
    report = ExpertReport(
        expert="MoleculeExpert",
        round=2,
        findings=[],
        confidence=ConfidenceInterval(low=0.5, high=0.7, sources=[]),
        delegations=[],
        data_gaps=[],
        bias_notes=[],
    )
    assert report.expert == "MoleculeExpert"
    assert report.round == 2
    assert report.confidence.low == 0.5


def test_delegation_added_to_report():
    delegation = TaskDelegation(
        target_expert="BioinfoExpert",
        record_ids=["rec_003"],
        reason="RNA-seq data requires bioinformatics assessment",
    )
    report = ExpertReport(
        expert="BiomedExpert",
        round=1,
        findings=[],
        confidence=ConfidenceInterval(low=0.0, high=1.0, sources=["no_data"]),
        delegations=[delegation],
        data_gaps=["No in vivo efficacy data for this target"],
        bias_notes=["All available studies from single lab"],
    )
    assert len(report.delegations) == 1
    assert report.delegations[0].target_expert == "BioinfoExpert"


def test_final_report_structure():
    fr = FinalReport(
        drug_id="DRUG_A",
        disease_id="DISEASE_X",
        endpoint="primary_endpoint_change",
        efficacy_low=0.2,
        efficacy_up=0.8,
        confidence_level="moderate",
        reasoning_mode="Iris-expert",
        expert_consensus="Multiple experts agree on moderate efficacy signal",
        key_findings=[
            "MoleculeExpert: strong target binding",
            "ClinicExpert: limited phase 2 data",
        ],
        contradictions=["BiomedExpert notes conflicting animal model results"],
        data_gaps=["No phase 3 trial data"],
        supporting_records=["rec_001", "rec_005"],
        per_expert_reports={},
    )
    assert fr.drug_id == "DRUG_A"
    assert fr.efficacy_low == 0.2
    assert fr.efficacy_up == 0.8
    assert len(fr.contradictions) == 1


def test_expert_context_carries_guidance():
    ctx = ExpertContext(
        drug_ids=["DRUG_A"],
        disease_id="DISEASE_X",
        endpoints=["primary_endpoint_change"],
        round=2,
        guidance="Focus on contradiction between molecular binding and cellular efficacy data",
        history=[],
    )
    assert ctx.guidance is not None
    assert ctx.round == 2
