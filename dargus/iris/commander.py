"""Iris commander — orchestrates global D-Base workflows."""

from __future__ import annotations

import logging
from typing import Any, Callable

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.dbase.paths import default_dargus_home
from dargus.experts.disease import DiseaseExpert
from dargus.iris.analog import IrisAnalog
from dargus.iris.bayes import IrisBayes
from dargus.iris.ensemble import IrisEnsemble
from dargus.iris.expert import IrisExpert
from dargus.iris.gnn import IrisGnn
from dargus.iris.llm import IrisLlm
from dargus.iris.search import IrisSearch
from dargus.workflows.train import TrainingReport
from dargus.workflows.train import run as run_train

logger = logging.getLogger(__name__)


class Iris:
    """Coordinates D-Base, expert system, and Iris-* agents."""

    name = "Iris"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def _global_manager(self) -> DBaseManager:
        dbase = DBase.global_instance()
        self._ensure_default_templates(dbase)
        return DBaseManager(dbase)

    def status(self) -> dict[str, Any]:
        """Report global D-Base status."""
        dbase = DBase.global_instance()
        return {
            "dargus_home": str(default_dargus_home()),
            "dbase_dir": str(dbase.dbase_dir),
            "n_records": len(dbase.list_records()),
            "n_templates": len(dbase._templates),
        }

    def train(self, datadir: str) -> TrainingReport:
        """Run the Train workflow on the global D-Base."""
        return run_train(datadir)

    def infer(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
        datadir: str | None = None,
        confirm_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run the Infer workflow.

        ``confirm_callback`` receives the ``PlanProposal`` for approval before
        agents execute. If ``datadir`` is provided, training has already run by
        the workflow layer; this method performs prediction only.
        """
        manager = self._global_manager()
        disease_expert = DiseaseExpert(manager)

        plan = disease_expert.plan(drug_ids, disease_id, endpoints)
        if confirm_callback is not None and not confirm_callback(plan.to_dict()):
            return {
                "aborted": True,
                "reason": "Prediction plan was not confirmed by user.",
                "plan_suggestion": plan.to_dict(),
            }

        predictions: dict[str, dict[str, dict[str, Any]]] = {}
        for agent_name in plan.agents:
            try:
                agent_predictions = self._run_agent(
                    agent_name, manager, drug_ids, disease_id, plan.endpoints
                )
                if agent_predictions:
                    predictions[agent_name] = agent_predictions
            except Exception:
                logger.warning(
                    "Agent %s failed during prediction, skipping", agent_name, exc_info=True
                )

        if not predictions:
            return {
                drug: {endpoint: self._empty_pred() for endpoint in plan.endpoints}
                for drug in drug_ids
            }
        if len(predictions) == 1:
            return next(iter(predictions.values()))
        return self.ensemble(predictions, plan.weights)

    def benchmark(self, config_path: str) -> dict[str, Any]:
        """Run the Benchmark workflow."""
        from dargus.workflows.benchmark import run as run_benchmark

        return run_benchmark(config_path)

    def _run_agent(
        self,
        name: str,
        manager: DBaseManager,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
    ) -> dict[str, dict[str, dict[str, Any]]] | None:
        dbase = manager.dbase
        if name == "Iris-expert":
            agent = IrisExpert(DiseaseExpert(manager))
            return agent.predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-search":
            return IrisSearch().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-analog":
            return IrisAnalog().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-bayes":
            return IrisBayes().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-gnn":
            return IrisGnn().predict(dbase, drug_ids, disease_id, endpoints)
        if name == "Iris-llm":
            config = self.config if self.config else None
            return IrisLlm(config=config).predict(dbase, drug_ids, disease_id, endpoints)
        return None

    def _empty_pred(self) -> dict[str, Any]:
        return {
            "efficacy_low": 0.0,
            "efficacy_up": 1.0,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }

    def ensemble(
        self,
        predictions: dict[str, dict[str, dict[str, Any]]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Aggregate predictions from multiple Iris-* agents."""
        ensemble = IrisEnsemble()
        aggregated = ensemble.aggregate(predictions)
        if weights:
            for drug in aggregated:
                for endpoint in aggregated[drug]:
                    entry = aggregated[drug][endpoint]
                    components = entry.get("component_predictions", {})
                    weighted_low: list[float] = []
                    weighted_up: list[float] = []
                    denom = 0.0
                    for mode, interval in components.items():
                        w = weights.get(mode, 1.0)
                        weighted_low.append(interval["efficacy_low"] * w)
                        weighted_up.append(interval["efficacy_up"] * w)
                        denom += w
                    if denom > 0:
                        entry["efficacy_low"] = sum(weighted_low) / denom
                        entry["efficacy_up"] = sum(weighted_up) / denom
        return aggregated

    def _ensure_default_templates(self, dbase: DBase) -> None:
        drug_vocab = "global_drug_vocab"
        disease_vocab = "global_disease_vocab"
        endpoint_vocab = "global_endpoint_vocab"
        if "clinical_trial_outcome_v1" not in dbase._templates:
            from dargus.dbase import TemplateSchema

            dbase.add_template(
                TemplateSchema(
                    template_id="clinical_trial_outcome_v1",
                    fields=[
                        {
                            "name": "biological_level",
                            "type": "factor",
                            "vocabulary": [
                                "molecular",
                                "cellular",
                                "exvivo",
                                "animal",
                                "clinical",
                                "epi",
                            ],
                        },
                        {"name": "drug_id", "type": "factor", "vocabulary_ref": drug_vocab},
                        {"name": "disease_id", "type": "factor", "vocabulary_ref": disease_vocab},
                        {"name": "endpoint", "type": "factor", "vocabulary_ref": endpoint_vocab},
                        {"name": "fold_change", "type": "float"},
                        {"name": "ci95_lower", "type": "float"},
                        {"name": "ci95_upper", "type": "float"},
                    ],
                )
            )
