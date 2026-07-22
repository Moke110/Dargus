"""DEPRECATED: Iris-* expert agent wrapper (v0.15.0)."""

from __future__ import annotations

import warnings
from typing import Any

from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisExpert(IrisAgent):
    """DEPRECATED: Use ``dargus.iris.commander.Iris.predict()`` instead.
    Kept for backward compatibility. Will be removed.
    """

    name = "Iris-expert"

    def __init__(self, disease_expert=None):
        warnings.warn(
            "dargus.iris.expert.IrisExpert is deprecated, "
            "use dargus.iris.commander.Iris.predict() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.disease_expert = disease_expert

    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        # Route to Iris commander
        from dargus.iris.commander import Iris

        iris = Iris()
        result: PredictionMatrix = {}
        for drug_id in drug_ids:
            result[drug_id] = {}
            preds = iris.predict(
                drug_ids=[drug_id],
                disease_id=disease_id,
                endpoints=endpoints,
            )
            # preds: {drug: {disease: {endpoint: {...}}}}
            for _disease, eps in preds.get(drug_id, {}).items():
                for epid, pred in eps.items():
                    result[drug_id][epid] = {
                        "efficacy_low": pred.get("efficacy_low", 0.0),
                        "efficacy_up": pred.get("efficacy_up", 1.0),
                        "confidence_level": pred.get("confidence_level", "unknown"),
                    "reasoning_mode": self.name,
                    "supporting_records": pred.get("supporting_records", []),
                }
        return result
