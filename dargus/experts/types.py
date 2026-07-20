"""Dargus expert system types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurateResult:
    """Result of curation: accepted records plus metadata about rejected ones."""

    records: list[dict[str, Any]]
    misclassified: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanProposal:
    """Prediction plan produced by DiseaseExpert."""

    drug_ids: list[str]
    disease_id: str
    endpoints: list[str]
    level_experts: list[str]
    agents: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_ids": self.drug_ids,
            "disease_id": self.disease_id,
            "endpoints": self.endpoints,
            "level_experts": self.level_experts,
            "agents": self.agents,
            "weights": self.weights,
            "reasoning": self.reasoning,
        }


@dataclass
class AnalysisReport:
    """Level-specific analysis report."""

    level: str
    summary: str
    records: list[str]
    contradictions: list[str] = field(default_factory=list)
    confidence: str = "low"


@dataclass
class IngestionResult:
    """Result of ingesting files into the expert system."""

    n_records: int
    files: list[str]
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractedInstance:
    """One raw instance extracted from a source file, before templating."""

    template_id: str
    raw_fields: dict[str, Any]
    source_file: str
    source_row: int
    extraction_confidence: str = "medium"


@dataclass
class ExtractionReport:
    """Result of one LevelExpert scanning raw data."""

    level: str
    files_considered: list[str]
    files_selected: list[str]
    source_types: dict[str, int]
    instances: list[ExtractedInstance]
    notes: list[str] = field(default_factory=list)

    @property
    def n_instances(self) -> int:
        return len(self.instances)


@dataclass
class IngestionSummary:
    """Aggregated extraction reports across all levels."""

    per_level: dict[str, ExtractionReport]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_instances(self) -> int:
        return sum(r.n_instances for r in self.per_level.values())

    @property
    def template_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self.per_level.values():
            for inst in report.instances:
                counts[inst.template_id] = counts.get(inst.template_id, 0) + 1
        return counts
