"""FourDExpert — Disease & Drug Development Director."""

from __future__ import annotations

from typing import Any

from dargus.agents.skill_registry import SkillRegistry
from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ConfidenceInterval,
    ExpertContext,
    ExpertReport,
    FinalReport,
)
from dargus.models.reasoning import ReasoningLLM
from dargus.runtime.hooks import HookRegistry
from dargus.tools.registry import ToolRegistry


class FourDExpert(Expert):
    """Disease & Drug Development Director.

    Holds broad-but-shallow knowledge across the full drug development
    stack. Does NOT perform technical orchestration (that's Iris's job).
    Synthesizes multi-Expert findings into a final conclusion and provides
    discussion guidance between rounds.
    """

    name = "FourDExpert"
    PERMITTED_TOOLS = ["dbase_query", "pubmed_search"]
    PERMITTED_KNOWLEDGE = ["dbase", "disease_rag"]
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
    DELEGATION_RULES = {}

    # ------------------------------------------------------------------
    # Domain-expert registry for delegate_to_expert
    # ------------------------------------------------------------------
    _DOMAIN_EXPERT_MAP: dict[str, str] = {
        "molecular": "dargus.experts.molecule.MoleculeExpert",
        "biomedical": "dargus.experts.biomed.BiomedExpert",
        "bioinformatics": "dargus.experts.bioinfo.BioinfoExpert",
        "clinical": "dargus.experts.clinic.ClinicExpert",
    }

    # ------------------------------------------------------------------
    # Constructor — DI passthrough to Expert → BaseAgent
    # ------------------------------------------------------------------

    def __init__(
        self,
        dbase: Any = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        knowledge_retrievers: dict[str, Any] | None = None,
        hook_registry: HookRegistry | None = None,
        agent_factory: Any | None = None,
    ):
        super().__init__(
            dbase=dbase,
            config=config,
            reasoning_llm=reasoning_llm,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            knowledge_retrievers=knowledge_retrievers,
            hook_registry=hook_registry,
        )
        self._agent_factory = agent_factory

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

    # ------------------------------------------------------------------
    # Coordination methods (Phase D)
    # ------------------------------------------------------------------

    def delegate_to_expert(
        self,
        domain: str,
        records: list[dict],
        question: str,
    ) -> dict[str, Any]:
        """Find/create the appropriate DomainExpert and delegate assessment.

        Args:
            domain: Domain key (``"molecular"``, ``"biomedical"``,
                ``"bioinformatics"``, ``"clinical"``).
            records: List of evidence records for the expert to assess.
            question: The assessment question to answer.

        Returns:
            Expert report dict with keys: ``domain``, ``conclusion``,
            ``confidence``, ``supporting_evidence``.

        Raises:
            ValueError: If *domain* is not recognised.
        """
        expert_cls_path = self._DOMAIN_EXPERT_MAP.get(domain)
        if expert_cls_path is None:
            raise ValueError(
                f"Unknown domain {domain!r}. Known domains: " f"{list(self._DOMAIN_EXPERT_MAP)}"
            )

        if self._agent_factory is not None:
            expert = self._agent_factory.expert(domain)
        else:
            module_path, class_name = expert_cls_path.rsplit(".", 1)
            import importlib

            mod = importlib.import_module(module_path)
            expert_cls = getattr(mod, class_name)
            expert = expert_cls(dbase=self.dbase)
        ctx = ExpertContext(
            drug_ids=[],
            disease_id="",
            endpoints=[],
        )
        report = expert.assess(records, ctx)

        return {
            "domain": domain,
            "conclusion": (
                f"{len(report.findings)} evidence items assessed "
                f"(avg quality: "
                f"{sum(f.quality_score for f in report.findings) / len(report.findings):.2f}"
                f")"
                if report.findings
                else "assessment complete"
            ),
            "confidence": {
                "low": report.confidence.low if report.confidence else 0.0,
                "high": report.confidence.high if report.confidence else 1.0,
            },
            "supporting_evidence": [
                {"record_ids": f.record_ids, "quality": f.quality_score} for f in report.findings
            ],
        }

    def synthesize(self, expert_reports: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine multiple ExpertReports into a unified D4Report.

        Args:
            expert_reports: List of expert report dicts (as returned by
                :meth:`delegate_to_expert`).

        Returns:
            Dict with keys: ``overall_conclusion``, ``confidence``,
            ``expert_reports``, ``conflicts``.
        """
        confidences: list[float] = []
        conclusions: list[str] = []
        conflicts: list[str] = []

        for report in expert_reports:
            conf = report.get("confidence", {})
            if isinstance(conf, dict):
                confidences.append(conf.get("low", 0.0))
                confidences.append(conf.get("high", 0.0))
            conclusion = report.get("conclusion", "")
            if conclusion:
                conclusions.append(f"[{report.get('domain', 'unknown')}] {conclusion}")

        # Detect conflicts: if any two reports have widely diverging confidence
        if len(confidences) >= 4:
            low_vals = confidences[::2]
            high_vals = confidences[1::2]
            if max(low_vals) > 0.6 and min(high_vals) < 0.4:
                conflicts.append(
                    "Confidence divergence detected: some experts are highly confident "
                    "while others are uncertain — cross-validation recommended"
                )

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if avg_confidence > 0.6:
            confidence_level = "high"
        elif avg_confidence > 0.3:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        return {
            "overall_conclusion": "; ".join(conclusions) if conclusions else "no assessment",
            "confidence": confidence_level,
            "expert_reports": expert_reports,
            "conflicts": conflicts,
        }

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

        # Compute DES ± DCS (design/4.1): efficacy_score = midpoint of the
        # experts' confidence bounds, confidence_score = half-width (how wide
        # the uncertainty band is). No supporting evidence means we refuse to
        # fake precision: scores stay unset and the level is
        # ``insufficient_data``.
        if confidence_scores and supporting_records:
            efficacy_low = min(confidence_scores)
            efficacy_up = max(confidence_scores)
            efficacy_score: float | None = (efficacy_low + efficacy_up) / 2
            confidence_score: float | None = (efficacy_up - efficacy_low) / 2
        else:
            efficacy_score = None
            confidence_score = None

        if efficacy_score is None:
            confidence_level = "insufficient_data"
        else:
            assert confidence_score is not None
            avg_conf = 1.0 - confidence_score
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
            efficacy_score=efficacy_score,
            confidence_score=confidence_score,
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
