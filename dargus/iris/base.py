from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol

from dargus.iris.probability_utils import normalize_prediction_entry

PredictionMatrix = dict[str, dict[str, dict[str, Any]]]


__all__ = ["IrisAgent", "PredictionMatrix", "normalize_prediction_entry"]


class IrisAgent(Protocol):
    name: str

    @abstractmethod
    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix: ...
