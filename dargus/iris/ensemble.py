from __future__ import annotations

from typing import Any

import numpy as np

from dargus.dbase import DBase
from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisEnsemble(IrisAgent):
    """Aggregates predictions from multiple Iris-* agents."""

    name = "Iris-ensemble"

    def __init__(self, agents: list[IrisAgent] | None = None):
        self.agents = agents or []

    def predict(
        self,
        dbase: DBase,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        predictions = {
            agent.name: agent.predict(dbase, drug_ids, disease_id, endpoints, embeddings, context)
            for agent in self.agents
        }
        return self.aggregate(predictions)

    def aggregate(self, predictions: dict[str, PredictionMatrix]) -> PredictionMatrix:
        result: PredictionMatrix = {}
        # Collect all drug/endpoints
        drugs: set[str] = set()
        endpoints: set[str] = set()
        for pred in predictions.values():
            for drug, eps in pred.items():
                drugs.add(drug)
                endpoints.update(eps.keys())

        for drug in drugs:
            result[drug] = {}
            for endpoint in endpoints:
                estimates: list[dict[str, Any]] = []
                for mode, pred in predictions.items():
                    entry = pred.get(drug, {}).get(endpoint)
                    if entry:
                        estimates.append({**entry, "_mode": mode})

                if not estimates:
                    continue

                weights = []
                for e in estimates:
                    width = e.get("ci95_upper", 1.0) - e.get("ci95_lower", 0.0)
                    if width <= 0:
                        width = 1.0
                    weights.append(1.0 / width)

                weights = np.array(weights)
                mean_effect = float(
                    np.average(
                        [e["normalized_effect_size"] for e in estimates],
                        weights=weights,
                    )
                )
                weighted_var = float(
                    np.average(
                        [(e["normalized_effect_size"] - mean_effect) ** 2 for e in estimates],
                        weights=weights,
                    )
                )
                ci_width = 1.96 * np.sqrt(weighted_var)

                all_supporting: list[str] = []
                for e in estimates:
                    all_supporting.extend(e.get("supporting_records", []))

                result[drug][endpoint] = {
                    "normalized_effect_size": mean_effect,
                    "ci95_lower": mean_effect - ci_width,
                    "ci95_upper": mean_effect + ci_width,
                    "supporting_records": sorted(set(all_supporting)),
                    "reasoning_mode": self.name,
                    "confidence_level": "ensemble",
                    "component_predictions": {
                        e["_mode"]: e["normalized_effect_size"] for e in estimates
                    },
                }
        return result
