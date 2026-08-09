"""BiomedExpert — preclinical biology evidence assessment."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
)


class BiomedExpert(Expert):
    """Assesses preclinical wet-lab evidence: cell assays, organoids,
    organ-on-chip, ex-vivo tissue, and animal studies."""

    name = "BiomedExpert"
    PERMITTED_TOOLS = ["dbase_query", "pubmed_search"]
    SUPPORTED_SKILLS = []

    SUPPORTED_LEVELS = (
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
    )
    SIM_PENALTY = 0.2

    def _collect_gaps(
        self, records: list[dict], findings: list[EvidenceAssessment]
    ) -> tuple[list[str], list[str]]:
        in_vivo_count = 0
        in_vitro_count = 0
        for finding in findings:
            if finding.biological_level in ("animal", "animal-sim"):
                in_vivo_count += 1
            else:
                in_vitro_count += 1

        data_gaps: list[str] = []
        if in_vivo_count == 0 and in_vitro_count > 0:
            data_gaps.append("No in vivo evidence — efficacy may not translate to organism level")
        return data_gaps, []

    def _assess_quality(self, record: dict) -> float:
        score = 0.5
        if self._read_field(record, "readout_value") is not None:
            score += 0.2
        level = self._read_biological_level(record)
        if level in ("animal", "animal-sim"):
            score += 0.15
        return min(max(score, 0.0), 1.0)

    def _assess_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_preclinical_evidence"])
        avg = sum(f.quality_score for f in findings) / len(findings)
        sources: list[str] = []
        sim = sum(1 for f in findings if "-sim" in f.biological_level)
        if sim > 0:
            sources.append("simulated_data_present")
        return ConfidenceInterval(
            low=max(0.0, avg - 0.2), high=min(1.0, avg + 0.2), sources=sources
        )
