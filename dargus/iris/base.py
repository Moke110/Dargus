from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

PredictionMatrix = dict[str, dict[str, dict[str, Any]]]


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
    ) -> PredictionMatrix:
        ...
