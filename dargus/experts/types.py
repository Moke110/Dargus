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
