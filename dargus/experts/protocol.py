"""v0.9.0 Expert protocol types — communication contract between Experts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceAssessment:
    """Assessment of one or more D-Base records by an Expert."""

    record_ids: list[str]
    biological_level: str
    relevance: str = "medium"
    quality_score: float = 0.5
    limitations: list[str] = field(default_factory=list)


@dataclass
class TaskDelegation:
    """Request to hand off records to another Expert."""

    target_expert: str
    record_ids: list[str]
    reason: str
    priority: str = "medium"


@dataclass
class ConfidenceInterval:
    """Numerical confidence interval with source attribution."""

    low: float
    high: float
    sources: list[str] = field(default_factory=list)


@dataclass
class ExpertReport:
    """Unified output from one Expert for one round."""

    expert: str
    round: int
    findings: list[EvidenceAssessment]
    confidence: ConfidenceInterval
    delegations: list[TaskDelegation] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    bias_notes: list[str] = field(default_factory=list)


@dataclass
class FinalReport:
    """Final synthesized report from 4DExpert after convergence.

    Scores follow the DES ± DCS contract (design/4.1): ``efficacy_score``
    (DES) and ``confidence_score`` (DCS), both 0–1 — or both ``None`` when
    ``confidence_level`` is ``"insufficient_data"`` (e.g. no supporting
    evidence in D-Base).
    """

    drug_id: str
    disease_id: str
    endpoint: str

    efficacy_score: float | None
    confidence_score: float | None
    confidence_level: str
    reasoning_mode: str = "Iris-expert"

    expert_consensus: str = ""
    key_findings: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    supporting_records: list[str] = field(default_factory=list)

    per_expert_reports: dict[str, list[ExpertReport]] = field(default_factory=dict)


@dataclass
class ExpertContext:
    """Context passed to each Expert.assess() call."""

    drug_ids: list[str]
    disease_id: str
    endpoints: list[str]
    round: int = 0
    guidance: str | None = None
    history: list[ExpertReport] = field(default_factory=list)
