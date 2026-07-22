"""ClinicExpert — rct and epidemiological evidence assessment (v0.15.0)."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    TaskDelegation,
)

_PHASE_WEIGHTS = {
    "phase_1": 0.3,
    "phase_2": 0.5,
    "phase_3": 0.7,
    "phase_4": 0.65,
    "rwe": 0.4,
    "meta_analysis": 0.8,
}


class ClinicExpert(Expert):
    """Assesses RCT, epidemiological, and post-market evidence.

    Covers rct, epi, and rct-sim levels with knowledge of trial design,
    medical statistics, and pharmacovigilance.
    """

    SUPPORTED_LEVELS = ("rct", "epi", "rct-sim")
    DELEGATION_RULES = {
        "molecular": "MoleculeExpert",
        "molecular-sim": "MoleculeExpert",
        "cellular": "BiomedExpert",
        "cellular-sim": "BiomedExpert",
        "exvivo": "BiomedExpert",
        "exvivo-sim": "BiomedExpert",
        "animal": "BiomedExpert",
        "animal-sim": "BiomedExpert",
    }

    def assess(
        self,
        records: list[dict],
        context: ExpertContext,
    ) -> ExpertReport:
        findings: list[EvidenceAssessment] = []
        delegations: list[TaskDelegation] = []
        data_gaps: list[str] = []
        bias_notes: list[str] = []

        for record in records:
            eid = record.get("evidence_id", "")
            level = self._read_biological_level(record)
            if level is None:
                continue

            if not self.can_handle(record):
                target = self.delegate_target(record)
                if target:
                    delegations.append(
                        TaskDelegation(
                            target_expert=target,
                            record_ids=[eid],
                            reason=f"Record level '{level}' outside ClinicExpert scope",
                        )
                    )
                continue

            quality = self._assess_quality(record)
            if "-sim" in (level or ""):
                bias_notes.append(f"Record {eid}: rct simulation data — no actual patient evidence")
                quality = max(0.0, quality - 0.3)

            findings.append(
                EvidenceAssessment(
                    record_ids=[eid],
                    biological_level=level or "unknown",
                    relevance="high" if level in ("rct", "epi") else "medium",
                    quality_score=quality,
                    limitations=[],
                )
            )

        # Mixed direction detection
        readouts = []
        for record in records:
            r = self._read_field(record, "readout_value")
            if r is not None:
                try:
                    readouts.append(float(r))
                except (TypeError, ValueError):
                    pass
        if readouts and any(r > 0 for r in readouts) and any(r < 0 for r in readouts):
            bias_notes.append("Mixed clinical effect directions detected")

        real_clinical = sum(1 for f in findings if f.biological_level in ("rct", "epi"))
        if real_clinical == 0:
            data_gaps.append("No real clinical trial evidence — only simulated or none")

        confidence = self._compute_confidence(findings)
        return ExpertReport(
            expert="ClinicExpert",
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=data_gaps,
            bias_notes=bias_notes,
        )

    def _assess_quality(self, record: dict) -> float:
        score = 0.5
        phase = self._read_field(record, "phase")
        if phase is not None:
            phase_key = str(phase).strip().lower().replace(" ", "_")
            score = _PHASE_WEIGHTS.get(phase_key, 0.5)
        if self._read_field(record, "readout_value") is not None:
            score = min(score + 0.1, 1.0)
        return min(max(score, 0.0), 1.0)

    def _compute_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_clinical_evidence"])
        avg = sum(f.quality_score for f in findings) / len(findings)
        real_clinical = sum(1 for f in findings if f.biological_level in ("rct", "epi"))
        sources: list[str] = []
        if real_clinical == 0:
            sources.append("no_real_clinical_data")
        return ConfidenceInterval(
            low=max(0.0, avg - 0.15),
            high=min(1.0, avg + 0.15),
            sources=sources,
        )
