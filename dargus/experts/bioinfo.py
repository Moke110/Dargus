"""BioinfoExpert — high-throughput/omics evidence assessment."""

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
    """Assesses high-throughput and omics data across all biological levels.

    Unlike other Experts, BioinfoExpert is defined by methodology (high-throughput)
    rather than biological level. It only handles omics-scale data and delegates
    non-high-throughput records to the level-appropriate Expert.
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
        "clinical",
        "clinical-sim",
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
        "clinical": "ClinicExpert",
        "clinical-sim": "ClinicExpert",
    }

    def can_handle(self, record: TemplateRecord) -> bool:
        level = self._read_biological_level(record)
        if level not in self.SUPPORTED_LEVELS:
            return False
        return self._is_high_throughput(record)

    def _is_high_throughput(self, record: TemplateRecord) -> bool:
        assay = self._read_field(record, "assay_type")
        if assay is None:
            return False
        assay_lower = str(assay).lower().replace(" ", "_").replace("-", "_")
        return assay_lower in _HIGH_THROUGHPUT_ASSAYS

    def assess(
        self,
        records: list[TemplateRecord],
        context: ExpertContext,
    ) -> ExpertReport:
        findings: list[EvidenceAssessment] = []
        delegations: list[TaskDelegation] = []
        bias_notes: list[str] = []

        for record in records:
            level = self._read_biological_level(record)
            if level is None:
                continue

            if not self._is_high_throughput(record):
                target = self.delegate_target(record)
                if target:
                    delegations.append(
                        TaskDelegation(
                            target_expert=target,
                            record_ids=[record.record_id],
                            reason=f"Non-high-throughput data delegated to {target}",
                        )
                    )
                continue

            quality = self._assess_quality(record)
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
            expert="BioinfoExpert",
            round=context.round,
            findings=findings,
            confidence=confidence,
            delegations=delegations,
            data_gaps=[],
            bias_notes=bias_notes,
        )

    def _assess_quality(self, record: TemplateRecord) -> float:
        readout = self._read_field(record, "readout")
        log_pvalue = self._read_field(record, "log_pvalue")
        score = 0.5
        if readout is not None:
            score += 0.1
        if log_pvalue is not None:
            score += 0.2
        return min(max(score, 0.0), 1.0)

    def _compute_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_bioinfo_evidence"])
        avg = sum(f.quality_score for f in findings) / len(findings)
        return ConfidenceInterval(
            low=max(0.0, avg - 0.25), high=min(1.0, avg + 0.25), sources=[]
        )
