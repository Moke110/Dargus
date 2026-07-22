"""FourDExpert — Disease & Drug Development Director."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    ExpertReport,
    FinalReport,
)


class FourDExpert(Expert):
    """Disease & Drug Development Director.

    Holds broad-but-shallow knowledge across the full drug development
    stack. Does NOT perform technical orchestration (that's IrisExpert's job).
    Synthesizes multi-Expert findings into a final conclusion and provides
    discussion guidance between rounds.
    """

    SUPPORTED_LEVELS = (
        "molecular",
        "molecular-sim",
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
        "rct",
        "epi",
        "rct-sim",
    )
    DELEGATION_RULES = {}

    def assess(self, records, context):
        """FourDExpert does not assess individual records directly."""
        return ExpertReport(
            expert="FourDExpert",
            round=context.round,
            findings=[],
            confidence=ConfidenceInterval(low=0.0, high=1.0, sources=[]),
            delegations=[],
            data_gaps=[],
            bias_notes=[],
        )

    def conclude(
        self,
        drug_id: str,
        disease_id: str,
        endpoint: str,
        all_reports: dict[str, list[ExpertReport]],
    ) -> FinalReport:
        """Synthesize all Expert reports into a final conclusion."""
        all_findings = []
        contradictions: list[str] = []
        data_gaps: list[str] = []
        key_findings: list[str] = []
        supporting_records: list[str] = []
        confidence_scores: list[float] = []

        for expert_name, reports in all_reports.items():
            if expert_name == "FourDExpert":
                continue
            for report in reports:
                for finding in report.findings:
                    all_findings.append(finding)
                    supporting_records.extend(finding.record_ids)
                for note in report.bias_notes:
                    key_findings.append(f"[{expert_name}] {note}")
                if report.data_gaps:
                    data_gaps.extend(report.data_gaps)
                confidence_scores.append(report.confidence.low)
                confidence_scores.append(report.confidence.high)

        # Detect cross-Expert contradictions
        bio_positive = any(
            "positive" in str(n).lower() or "efficacy" in str(n).lower()
            for r in all_reports.get("BiomedExpert", [])
            for n in r.bias_notes
        )
        clinic_negative = any(
            "failed" in str(n).lower() or "negative" in str(n).lower()
            for r in all_reports.get("ClinicExpert", [])
            for n in r.bias_notes
        )
        if bio_positive and clinic_negative:
            contradictions.append(
                "Preclinical efficacy signal conflicts with clinical trial outcomes"
            )

        # Compute efficacy range
        if confidence_scores:
            efficacy_low = min(confidence_scores)
            efficacy_up = max(confidence_scores)
            avg_conf = sum(confidence_scores) / len(confidence_scores)
        else:
            efficacy_low = 0.0
            efficacy_up = 1.0
            avg_conf = 0.0

        if avg_conf > 0.6:
            confidence_level = "high"
        elif avg_conf > 0.3:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        # Consensus summary
        expert_names = [n for n in all_reports if n != "FourDExpert"]
        consensus = (
            f"{len(expert_names)} experts assessed {len(all_findings)} evidence items. "
            f"Overall confidence: {confidence_level}."
        )

        return FinalReport(
            drug_id=drug_id,
            disease_id=disease_id,
            endpoint=endpoint,
            efficacy_low=efficacy_low,
            efficacy_up=efficacy_up,
            confidence_level=confidence_level,
            reasoning_mode="Iris-expert",
            expert_consensus=consensus,
            key_findings=key_findings,
            contradictions=contradictions,
            data_gaps=data_gaps,
            supporting_records=supporting_records,
            per_expert_reports=all_reports,
        )

    def generate_guidance(self, reports: list[ExpertReport], round_num: int) -> str:
        """Generate discussion guidance for the next round based on current reports."""
        gaps: list[str] = []
        for report in reports:
            gaps.extend(report.data_gaps)

        sim_reports = [r for r in reports if any("-sim" in f.biological_level for f in r.findings)]

        parts: list[str] = []
        if gaps:
            parts.append(f"Address data gaps: {', '.join(gaps[:3])}")
        if sim_reports:
            parts.append(
                f"Verify findings from {len(sim_reports)} simulation-based report(s) "
                f"against real experimental evidence"
            )
        if not parts:
            parts.append("Continue cross-validation of findings across experts")

        return f"Round {round_num + 1} guidance: " + "; ".join(parts)
