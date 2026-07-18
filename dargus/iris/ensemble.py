from __future__ import annotations

from typing import Any

from dargus.iris.base import PredictionMatrix


class IrisEnsemble:
    """Aggregates predictions from multiple Iris-* agents."""

    name = "Iris-ensemble"

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

                total_weight = sum(weights)
                mean_effect = (
                    sum(e["normalized_effect_size"] * w for e, w in zip(estimates, weights))
                    / total_weight
                )

                # Ensemble CI: weighted average of widths around mean
                all_supporting: list[str] = []
                for e in estimates:
                    all_supporting.extend(e.get("supporting_records", []))

                result[drug][endpoint] = {
                    "normalized_effect_size": mean_effect,
                    "ci95_lower": min(e.get("ci95_lower", mean_effect) for e in estimates),
                    "ci95_upper": max(e.get("ci95_upper", mean_effect) for e in estimates),
                    "supporting_records": sorted(set(all_supporting)),
                    "reasoning_mode": self.name,
                    "confidence_level": "ensemble",
                    "component_predictions": {
                        e["_mode"]: e["normalized_effect_size"] for e in estimates
                    },
                }
        return result
