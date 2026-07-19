from __future__ import annotations

from typing import Any

from dargus.dbase.manager import DBaseManager
from dargus.experts.disease import DiseaseExpert
from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisExpert(IrisAgent):
    """Iris-* agent that wraps the DiseaseExpert expert system."""

    name = "Iris-expert"

    def __init__(self, disease_expert: DiseaseExpert | None = None):
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
        disease_expert = self.disease_expert
        if disease_expert is None:
            manager = DBaseManager(dbase)
            disease_expert = DiseaseExpert(manager)

        predictions = disease_expert.predict(
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints,
        )
        # Ensure canonical Iris-* keys and override reasoning mode.
        for drug in predictions:
            for endpoint in predictions[drug]:
                entry = predictions[drug][endpoint]
                entry["reasoning_mode"] = self.name
        return predictions
