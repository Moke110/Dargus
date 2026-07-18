from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.base import PredictionMatrix
from dargus.iris.ensemble import IrisEnsemble
from dargus.iris.llm import IrisLlm
from dargus.iris.search import IrisSearch


class IrisSelector:
    """Selects and runs Iris-* agents based on D-Base richness."""

    def __init__(self, dbase: DBase, config: dict[str, Any] | None = None):
        self.dbase = dbase
        self.config = config or {}

    def predict(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
    ) -> PredictionMatrix:
        if endpoints is None:
            endpoints = self._default_endpoints(disease_id)

        selected = self._select_agents(drug_ids, disease_id)
        predictions: dict[str, PredictionMatrix] = {}
        for agent in selected:
            predictions[agent.name] = agent.predict(
                self.dbase, drug_ids, disease_id, endpoints
            )

        ensemble = IrisEnsemble()
        return ensemble.aggregate(predictions)

    def _select_agents(self, drug_ids: list[str], disease_id: str) -> list[Any]:
        agents: list[Any] = []
        has_direct = False
        for drug in drug_ids:
            records = self.dbase.query(drug_id=drug, disease_id=disease_id)
            # Direct clinical records
            clinical = [
                r for r in records
                if self._record_level(r) == "clinical"
            ]
            if clinical:
                has_direct = True
            if records:
                agents.append(IrisSearch())
                break

        if has_direct:
            return [IrisSearch()]

        # Fallback: search + llm
        return [IrisSearch(), IrisLlm(config=self.config)]

    def _record_level(self, record: Any) -> str | None:
        schema = self.dbase._templates.get(record.template_id)
        if schema is None:
            return None
        try:
            idx = schema.field_index("biological_level")
        except KeyError:
            return None
        indices = record.sparse_vector.get("indices", [])
        values = record.sparse_vector.get("values", [])
        if idx not in indices:
            return None
        pos = indices.index(idx)
        factor_int = int(values[pos])
        # Reverse lookup in schema field vocabulary first
        field = schema.field_def("biological_level")
        if field.vocabulary and factor_int < len(field.vocabulary):
            return field.vocabulary[factor_int]
        # Fallback to global vocab
        vocab = self.dbase.vocab._vocab.get("biological_level", {})
        for term, val in vocab.items():
            if val == factor_int:
                return term
        return None

    def _default_endpoints(self, disease_id: str) -> list[str]:
        defaults = {
            "Parkinson's disease": ["UPDRS-III_change"],
            "Alzheimer's disease": ["ADAS-Cog_change"],
        }
        return defaults.get(disease_id, ["primary_endpoint_change"])
