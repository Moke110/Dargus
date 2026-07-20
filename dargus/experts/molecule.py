"""MoleculeExpert — molecular-level evidence assessment."""

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


class MoleculeExpert(Expert):
    """Assesses drug physicochemical properties, drug-target relationships,
    medicinal chemistry, and formulation evidence at the molecular level."""

    SUPPORTED_LEVELS = ("molecular", "molecular-sim")
    DELEGATION_RULES = {
        "cellular": "BiomedExpert",
        "cellular-sim": "BiomedExpert",
        "exvivo": "BiomedExpert",
        "exvivo-sim": "BiomedExpert",
        "animal": "BiomedExpert",
        "animal-sim": "BiomedExpert",
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
                            reason=f"Record level '{level}' outside MoleculeExpert scope",
                        )
                    )
                continue

            quality = self._assess_quality(record)
            if "-sim" in (level or ""):
                bias_notes.append(
                    f"Record {record.record_id}: computational/simulation data "
                    f"({level}) — lower evidential weight"
                )
                quality = max(0.0, quality - 0.2)

            findings.append(
                EvidenceAssessment(
                    record_ids=[record.record_id],
                    biological_level=level or "unknown",
                    relevance="medium",
                    quality_score=quality,
                    limitations=[],
                )
            )

        confidence = self._compute_confidence(findings)
        return ExpertReport(
            expert="MoleculeExpert",
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=data_gaps,
            bias_notes=bias_notes,
        )

    def _assess_quality(self, record: TemplateRecord) -> float:
        readout = self._read_field(record, "readout")
        has_readout = readout is not None
        target_id = self._read_field(record, "target_id")
        has_target = target_id is not None

        score = 0.5
        if has_readout:
            score += 0.2
        if has_target:
            score += 0.1
        source = record.source
        if isinstance(source, dict) and source.get("type") == "auto_extract":
            score -= 0.1
        return min(max(score, 0.0), 1.0)

    def _compute_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
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
