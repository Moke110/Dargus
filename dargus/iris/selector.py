from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.analog import IrisAnalog
from dargus.iris.base import IrisAgent, PredictionMatrix
from dargus.iris.bayes import IrisBayes
from dargus.iris.ensemble import IrisEnsemble
from dargus.iris.gnn import IrisGnn
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

        agents = self.select(drug_ids, disease_id)
        ensemble = IrisEnsemble(agents)
        return ensemble.predict(
            self.dbase,
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints,
        )

    def select(self, drug_ids: list[str], disease_id: str) -> list[IrisAgent]:
        counts = self._count_records(drug_ids, disease_id)
        has_clinical = counts.get("clinical", 0) >= 1
        has_preclinical = (
            sum(counts.get(level, 0) for level in ["molecular", "cellular", "exvivo", "animal"])
            >= 2
        )
        total = sum(counts.values())

        agents: list[IrisAgent] = [IrisSearch()]
        if has_clinical:
            agents.append(IrisBayes())
        if has_preclinical:
            agents.extend([IrisAnalog(), IrisBayes()])
        if total <= 2:
            agents.append(IrisLlm(config=self.config))
        if total >= 10:
            agents.append(IrisGnn())
        return agents

    def _count_records(self, drug_ids: list[str], disease_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.dbase.query(disease_id=disease_id):
            schema = self.dbase._templates.get(rec.template_id)
            if schema is None:
                continue
            try:
                idx = schema.field_index("drug_id")
            except KeyError:
                continue
            indices = rec.sparse_vector.get("indices", [])
            values = rec.sparse_vector.get("values", [])
            if idx not in indices:
                continue
            val = int(values[indices.index(idx)])
            vocab_ref = schema.field_def("drug_id").vocabulary_ref or "drug_id"
            drug = self.dbase.vocab.reverse_lookup(vocab_ref, val)
            if drug not in drug_ids:
                continue

            level = self._record_level(rec) or "molecular"
            counts[level] = counts.get(level, 0) + 1
        return counts

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
        field = schema.field_def("biological_level")
        if field.vocabulary and factor_int < len(field.vocabulary):
            return field.vocabulary[factor_int]
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
