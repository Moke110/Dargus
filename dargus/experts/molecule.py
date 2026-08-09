"""MoleculeExpert — molecular-level evidence assessment."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
)


class MoleculeExpert(Expert):
    """Assesses drug physicochemical properties, drug-target relationships,
    medicinal chemistry, and formulation evidence at the molecular level."""

    name = "MoleculeExpert"
    PERMITTED_TOOLS = ["dbase_query", "pubmed_search"]
    SUPPORTED_SKILLS = ["dti_prediction", "admet_assessment", "molecular_similarity"]

    SUPPORTED_LEVELS = ("molecular", "molecular-sim")
    SIM_PENALTY = 0.2
    SIM_BIAS_MSG = "Record {eid}: computational/simulation data ({level}) — lower evidential weight"

    def _assess_quality(self, record: dict) -> float:
        has_readout = self._read_field(record, "readout_value") is not None
        score = 0.5
        if has_readout:
            score += 0.2
        # Check for target via interventions
        interventions = record.get("interventions", [])
        has_target = any(i.get("entity_type") == "gene" for i in interventions)
        if has_target:
            score += 0.1
        sources = record.get("sources", [])
        for s in sources:
            if s.get("type") == "auto_extract":
                score -= 0.1
                break
        return min(max(score, 0.0), 1.0)

    def _assess_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_molecular_evidence"])
        avg_quality = sum(f.quality_score for f in findings) / len(findings)
        sim_count = sum(1 for f in findings if "-sim" in f.biological_level)
        sources: list[str] = []
        if sim_count > 0:
            sources.append("simulated_data_present")
        low = max(0.0, avg_quality - 0.2)
        high = min(1.0, avg_quality + 0.2)
        return ConfidenceInterval(low=low, high=high, sources=sources)
