"""BiomedExpert — preclinical biology evidence assessment."""

from __future__ import annotations

from dargus.dbase import TemplateRecord
from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    TaskDelegation,
)


class BiomedExpert(Expert):
    """Assesses preclinical wet-lab evidence: cell assays, organoids,
    organ-on-chip, ex-vivo tissue, and animal studies."""

    SUPPORTED_LEVELS = (
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
    )
    DELEGATION_RULES = {
        "molecular": "MoleculeExpert",
        "molecular-sim": "MoleculeExpert",
        "clinical": "ClinicExpert",
        "clinical-sim": "ClinicExpert",
    }

    def assess(
        self,
        records: list[TemplateRecord],
        context: ExpertContext,
    ) -> ExpertReport:
        findings: list[EvidenceAssessment] = []
        delegations: list[TaskDelegation] = []
        data_gaps: list[str] = []
        bias_notes: list[str] = []

        in_vivo_count = 0
        in_vitro_count = 0

        for record in records:
            level = self._read_biological_level(record)
            if level is None:
                continue

            if not self.can_handle(record):
                target = self.delegate_target(record)
                if target:
                    delegations.append(
                        TaskDelegation(
                            target_expert=target,
                            record_ids=[record.record_id],
                            reason=f"Record level '{level}' outside BiomedExpert scope",
                        )
                    )
                continue

            quality = self._assess_quality(record)
            if "-sim" in (level or ""):
                bias_notes.append(f"Record {record.record_id}: simulation-derived data ({level})")
                quality = max(0.0, quality - 0.2)

            if level in ("animal", "animal-sim"):
                in_vivo_count += 1
            else:
                in_vitro_count += 1

            findings.append(
                EvidenceAssessment(
                    record_ids=[record.record_id],
                    biological_level=level or "unknown",
                    relevance="medium",
                    quality_score=quality,
                    limitations=[],
                )
            )

        if in_vivo_count == 0 and in_vitro_count > 0:
            data_gaps.append("No in vivo evidence — efficacy may not translate to organism level")

        confidence = self._compute_confidence(findings)
        return ExpertReport(
            expert="BiomedExpert",
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=data_gaps,
            bias_notes=bias_notes,
        )

    def _assess_quality(self, record: TemplateRecord) -> float:
        readout = self._read_field(record, "readout")
        score = 0.5
        if readout is not None:
            score += 0.2
        level = self._read_biological_level(record)
        if level in ("animal", "animal-sim"):
            score += 0.15
        return min(max(score, 0.0), 1.0)

    def _compute_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
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
