"""BioinfoExpert — high-throughput/omics evidence assessment."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
    ExpertContext,
    ExpertReport,
    TaskDelegation,
)

_HIGH_THROUGHPUT_ASSAYS = frozenset(
    {
        "rna_seq",
        "rnaseq",
        "microarray",
        "scrnaseq",
        "single_cell",
        "chip_seq",
        "atac_seq",
        "methylation",
        "proteomics",
        "metabolomics",
        "gwas",
        "twos",
        "eqtl",
        "pqtl",
        "wes",
        "wgs",
        "whole_exome",
        "whole_genome",
        "crispr_screen",
        "rnai_screen",
    }
)


class BioinfoExpert(Expert):
    """Assesses high-throughput and omics data across all biological levels."""

    name = "BioinfoExpert"
    PERMITTED_TOOLS = ["dbase_query", "pubmed_search"]
    SUPPORTED_SKILLS = []

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
    DELEGATION_RULES = {
        "molecular": "MoleculeExpert",
        "molecular-sim": "MoleculeExpert",
        "cellular": "BiomedExpert",
        "cellular-sim": "BiomedExpert",
        "exvivo": "BiomedExpert",
        "exvivo-sim": "BiomedExpert",
        "animal": "BiomedExpert",
        "animal-sim": "BiomedExpert",
        "rct": "ClinicExpert",
        "epi": "ClinicExpert",
        "rct-sim": "ClinicExpert",
    }

    def can_handle(self, record: dict) -> bool:
        level = self._read_biological_level(record)
        if level not in self.SUPPORTED_LEVELS:
            return False
        return self._is_high_throughput(record)

    def _is_high_throughput(self, record: dict) -> bool:
        platform = record.get("platform") or {}
        assay = platform.get("assay_platform") or record.get("assay_type")
        if assay is None:
            return False
        assay_lower = str(assay).lower().replace(" ", "_").replace("-", "_")
        return assay_lower in _HIGH_THROUGHPUT_ASSAYS

    def assess(
        self,
        records: list[dict],
        context: ExpertContext,
    ) -> ExpertReport:
        findings: list[EvidenceAssessment] = []
        delegations: list[TaskDelegation] = []
        bias_notes: list[str] = []

        for record in records:
            eid = record.get("evidence_id", "")
            level = self._read_biological_level(record)
            if level is None:
                continue

            if not self._is_high_throughput(record):
                target = self.delegate_target(record)
                if target:
                    delegations.append(
                        TaskDelegation(
                            target_expert=target,
                            record_ids=[eid],
                            reason=f"Non-high-throughput data delegated to {target}",
                        )
                    )
                continue

            quality = self._assess_quality(record)
            findings.append(
                EvidenceAssessment(
                    record_ids=[eid],
                    biological_level=level or "unknown",
                    relevance="medium",
                    quality_score=quality,
                    limitations=[],
                )
            )

        confidence = self._compute_confidence(findings)
        return ExpertReport(
            expert="BioinfoExpert",
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=[],
            bias_notes=bias_notes,
        )

    def _assess_quality(self, record: dict) -> float:
        score = 0.5
        if self._read_field(record, "readout_value") is not None:
            score += 0.1
        if self._read_field(record, "p_value") is not None:
            score += 0.2
        return min(max(score, 0.0), 1.0)

    def _compute_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_bioinfo_evidence"])
        avg = sum(f.quality_score for f in findings) / len(findings)
        return ConfidenceInterval(low=max(0.0, avg - 0.25), high=min(1.0, avg + 0.25), sources=[])
