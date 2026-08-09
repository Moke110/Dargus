"""BioinfoExpert — high-throughput/omics evidence assessment."""

from __future__ import annotations

from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    EvidenceAssessment,
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
    # Bioinfo applies no simulation penalty or bias note — sim-derived omics
    # records are assessed at face value.
    SIM_PENALTY = 0.0
    SIM_BIAS_MSG = ""

    def can_handle(self, record: dict) -> bool:
        level = self._read_biological_level(record)
        if level not in self.SUPPORTED_LEVELS:
            return False
        return self._is_high_throughput(record)

    def _gate(self, record: dict) -> bool:
        """High-throughput gating: admission is the assay test, not level scope.

        A high-throughput record at any biological level is assessed; anything
        else is delegated to the level's domain Expert.
        """
        return self._is_high_throughput(record)

    def _is_high_throughput(self, record: dict) -> bool:
        platform = record.get("platform") or {}
        assay = platform.get("assay_platform") or record.get("assay_type")
        if assay is None:
            return False
        assay_lower = str(assay).lower().replace(" ", "_").replace("-", "_")
        return assay_lower in _HIGH_THROUGHPUT_ASSAYS

    def _delegation_reason(self, level: str, target: str) -> str:
        return f"Non-high-throughput data delegated to {target}"

    def _assess_quality(self, record: dict) -> float:
        score = 0.5
        if self._read_field(record, "readout_value") is not None:
            score += 0.1
        if self._read_field(record, "p_value") is not None:
            score += 0.2
        return min(max(score, 0.0), 1.0)

    def _assess_confidence(self, findings: list[EvidenceAssessment]) -> ConfidenceInterval:
        if not findings:
            return ConfidenceInterval(low=0.0, high=1.0, sources=["no_bioinfo_evidence"])
        avg = sum(f.quality_score for f in findings) / len(findings)
        return ConfidenceInterval(low=max(0.0, avg - 0.25), high=min(1.0, avg + 0.25), sources=[])
