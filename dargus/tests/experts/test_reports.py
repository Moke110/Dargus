"""Tests for the shared ExpertReport serializer + expert domain registry (#92).

Locks in the prefactor contract: the serialization is a lossless round-trip,
the domain tables are mutually consistent, and the predict task_spec helper
produces the shape the spawn tool / Iris consume.
"""

from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertReport,
    FinalReport,
    TaskDelegation,
)
from dargus.experts.reports import (
    DOMAIN_EXPERT_PATHS,
    DOMAIN_EXPERTS,
    EXPERT_DOMAINS,
    EXPERT_NAME_TO_DOMAIN,
    expert_report_from_dict,
    expert_report_to_dict,
    final_report_from_dict,
    final_report_to_dict,
    predict_task_spec,
)


def _sample_report() -> ExpertReport:
    return ExpertReport(
        expert="MoleculeExpert",
        round=2,
        findings=[
            EvidenceAssessment(
                record_ids=["rec_001", "rec_002"],
                biological_level="molecular",
                relevance="high",
                quality_score=0.8,
                limitations=["single study"],
            )
        ],
        confidence=ConfidenceInterval(low=0.4, high=0.9, sources=["in_vitro"]),
        delegations=[
            TaskDelegation(
                target_expert="BioinfoExpert",
                record_ids=["rec_002"],
                reason="needs bioinformatics scope",
                priority="high",
            )
        ],
        data_gaps=["no in vivo data"],
        bias_notes=["small sample"],
    )


def test_expert_report_serializer_round_trip():
    """to_dict → from_dict recovers the full report field-for-field."""
    original = _sample_report()
    restored = expert_report_from_dict(expert_report_to_dict(original))

    assert restored.expert == original.expert
    assert restored.round == original.round
    assert restored.findings == original.findings
    assert restored.confidence == original.confidence
    assert restored.delegations == original.delegations
    assert restored.data_gaps == original.data_gaps
    assert restored.bias_notes == original.bias_notes


def test_expert_report_serializer_handles_empty():
    """A minimal report round-trips through the default-filled deserializer."""
    payload = {"expert": "ClinicExpert", "round": 0}
    restored = expert_report_from_dict(payload)
    assert restored.findings == []
    assert restored.confidence.low == 0.0
    assert restored.confidence.high == 1.0
    assert restored.delegations == []


def test_domain_registry_consistency():
    """The domain tables agree: EXPERT_DOMAINS ⊇ the domain keys, and the
    class-name alias map is the exact inverse of the path map."""
    assert set(DOMAIN_EXPERT_PATHS) == set(DOMAIN_EXPERTS)
    assert set(EXPERT_NAME_TO_DOMAIN.values()) == set(DOMAIN_EXPERTS)
    for domain, path in DOMAIN_EXPERT_PATHS.items():
        assert path.rsplit(".", 1)[1] in EXPERT_NAME_TO_DOMAIN
        assert EXPERT_NAME_TO_DOMAIN[path.rsplit(".", 1)[1]] == domain
    # The spawn tool's accepted keys include the D4 director on top.
    assert set(EXPERT_DOMAINS) == set(DOMAIN_EXPERTS) | {"d4"}


def test_predict_task_spec_shape():
    spec = predict_task_spec(drug="chembl:1", disease="MONDO:1", endpoint="IC50", session_id="s1")
    assert spec == {
        "workflow": "predict",
        "drug_ids": ["chembl:1"],
        "disease_id": "MONDO:1",
        "endpoints": ["IC50"],
        "session_id": "s1",
    }


def test_final_report_serializer_round_trip():
    """#96: a D4 FinalReport serializes into the universal contract dict and
    round-trips back field-for-field."""
    original = FinalReport(
        drug_id="chembl:1",
        disease_id="MONDO:1",
        endpoint="IC50",
        efficacy_score=0.6,
        confidence_score=0.2,
        confidence_level="moderate",
        reasoning_mode="Iris-expert",
        supporting_records=["rec_1", "rec_2"],
        expert_consensus="2 experts assessed 1 evidence items. Overall confidence: moderate.",
        contradictions=["biomed vs clinic"],
        data_gaps=["no phase 3"],
    )
    contract = final_report_to_dict(original)
    # The universal nested shape predict() returns.
    entry = contract["chembl:1"]["MONDO:1"]["IC50"]
    assert entry["efficacy_score"] == 0.6
    assert entry["confidence_level"] == "moderate"

    restored = final_report_from_dict(contract)
    assert restored == original
    assert restored.drug_id == original.drug_id
    assert restored.disease_id == original.disease_id
    assert restored.endpoint == original.endpoint
    assert restored.efficacy_score == original.efficacy_score
    assert restored.confidence_score == original.confidence_score
    assert restored.confidence_level == original.confidence_level
