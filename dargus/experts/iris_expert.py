"""IrisExpert — technical orchestration layer for the Expert system."""

from __future__ import annotations

from dargus.dbase import TemplateRecord
from dargus.experts.base import Expert
from dargus.experts.protocol import (
    ExpertContext,
    ExpertReport,
    FinalReport,
)


class IrisExpert:
    """Manages the round-based Expert dialog protocol.

    Distributes records, collects reports, handles delegations,
    judges convergence, and passes results to FourDExpert for conclusion.
    """

    def __init__(
        self,
        molecule=None,
        biomed=None,
        bioinfo=None,
        clinic=None,
        director=None,
    ):
        self.molecule = molecule
        self.biomed = biomed
        self.bioinfo = bioinfo
        self.clinic = clinic
        self.director = director
        self.max_rounds = 5
        self._record_cache: dict[str, TemplateRecord] = {}

    @property
    def _experts(self) -> dict[str, Expert]:
        experts: dict[str, Expert] = {}
        for name, attr in [
            ("MoleculeExpert", self.molecule),
            ("BiomedExpert", self.biomed),
            ("BioinfoExpert", self.bioinfo),
            ("ClinicExpert", self.clinic),
            ("FourDExpert", self.director),
        ]:
            if attr is not None:
                experts[name] = attr
        return experts

    def run(self, records: list[TemplateRecord], context: ExpertContext) -> FinalReport:
        """Execute the multi-round dialog protocol until convergence.

        Args:
            records: D-Base records to assess.
            context: ExpertContext with drug/disease/endpoint info.

        Returns:
            FinalReport synthesized by FourDExpert (or fallback).
        """
        for rec in records:
            self._record_cache[rec.record_id] = rec

        all_reports: dict[str, list[ExpertReport]] = {}
        all_seen: set[tuple[str, str]] = set()

        distribution = self._distribute(records)

        for round_num in range(1, self.max_rounds + 1):
            context.round = round_num
            if self.director and round_num > 1:
                prev_round_reports = []
                for reports in all_reports.values():
                    prev_round_reports.extend([r for r in reports if r.round == round_num - 1])
                context.guidance = self.director.generate_guidance(
                    prev_round_reports, round_num - 1
                )

            round_reports = self._run_round(distribution, context, all_seen)
            for expert_name, report in round_reports.items():
                all_reports.setdefault(expert_name, []).append(report)

            new_delegations = self._collect_fresh_delegations(round_reports, all_seen)
            if not new_delegations:
                break

            distribution = self._delegations_to_distribution(new_delegations)

        if self.director is None:
            return self._fallback_conclude(context, all_reports)

        return self.director.conclude(
            drug_id=context.drug_ids[0] if context.drug_ids else "unknown",
            disease_id=context.disease_id,
            endpoint=context.endpoints[0] if context.endpoints else "unknown",
            all_reports=all_reports,
        )

    def _run_round(
        self,
        distribution: dict[str, list[TemplateRecord]],
        context: ExpertContext,
        all_seen: set[tuple[str, str]],
    ) -> dict[str, ExpertReport]:
        reports: dict[str, ExpertReport] = {}
        for expert_name, recs in distribution.items():
            if not recs:
                continue
            expert = self._experts.get(expert_name)
            if expert is None:
                continue
            report = expert.assess(recs, context)
            reports[expert_name] = report
            for rec in recs:
                all_seen.add((expert_name, rec.record_id))
        return reports

    def _distribute(self, records: list[TemplateRecord]) -> dict[str, list[TemplateRecord]]:
        dist: dict[str, list[TemplateRecord]] = {}
        for rec in records:
            assigned = False
            for name, expert in self._experts.items():
                if name == "FourDExpert":
                    continue
                if expert.can_handle(rec):
                    dist.setdefault(name, []).append(rec)
                    assigned = True
                    break
            if not assigned:
                dist.setdefault("FourDExpert", []).append(rec)
        return dist

    def _collect_fresh_delegations(
        self,
        round_reports: dict[str, ExpertReport],
        all_seen: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        fresh: list[tuple[str, str]] = []
        for report in round_reports.values():
            for delegation in report.delegations:
                for rec_id in delegation.record_ids:
                    if (delegation.target_expert, rec_id) not in all_seen:
                        fresh.append((delegation.target_expert, rec_id))
        return fresh

    def _delegations_to_distribution(
        self, delegations: list[tuple[str, str]]
    ) -> dict[str, list[TemplateRecord]]:
        dist: dict[str, list[TemplateRecord]] = {}
        for expert_name, rec_id in delegations:
            record = self._record_cache.get(rec_id)
            if record is not None:
                dist.setdefault(expert_name, []).append(record)
        return dist

    def _fallback_conclude(
        self,
        context: ExpertContext,
        all_reports: dict[str, list[ExpertReport]],
    ) -> FinalReport:
        return FinalReport(
            drug_id=context.drug_ids[0] if context.drug_ids else "unknown",
            disease_id=context.disease_id,
            endpoint=context.endpoints[0] if context.endpoints else "unknown",
            efficacy_low=0.0,
            efficacy_up=1.0,
            confidence_level="low",
            reasoning_mode="Iris-expert",
            per_expert_reports=all_reports,
        )
