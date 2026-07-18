from __future__ import annotations

from typing import Any

import numpy as np

from dargus.dbase import DBase
from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisAnalog(IrisAgent):
    name = "Iris-analog"

    def predict(
        self,
        dbase: DBase,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        result: PredictionMatrix = {}
        for drug_id in drug_ids:
            result[drug_id] = {}
            analog_records = self._find_analogs(dbase, drug_id, disease_id)
            for endpoint in endpoints:
                if not analog_records:
                    result[drug_id][endpoint] = {
                        "normalized_effect_size": 0.0,
                        "ci95_lower": -1.0,
                        "ci95_upper": 1.0,
                        "supporting_records": [],
                        "reasoning_mode": self.name,
                        "confidence_level": "insufficient_data",
                    }
                    continue

                values = [r["value"] for r in analog_records]
                mean = float(np.mean(values))
                std = float(np.std(values)) if len(values) > 1 else 1.0
                result[drug_id][endpoint] = {
                    "normalized_effect_size": mean,
                    "ci95_lower": float(mean - 1.96 * std),
                    "ci95_upper": float(mean + 1.96 * std),
                    "supporting_records": [r["record_id"] for r in analog_records[:10]],
                    "reasoning_mode": self.name,
                    "confidence_level": "analogical_evidence",
                }
        return result

    def _find_analogs(self, dbase: DBase, drug_id: str, disease_id: str) -> list[dict]:
        analogs = []
        target_records = dbase.query()
        for rec in target_records:
            schema = dbase._templates.get(rec.template_id)
            if schema is None:
                continue
            rec_drug = self._factor_value(dbase, rec, schema, "drug_id")
            rec_disease = self._factor_value(dbase, rec, schema, "disease_id")
            if rec_drug == drug_id and rec_disease == disease_id:
                val = self._read_value(dbase, rec, schema)
                if val is not None:
                    analogs.append({"record_id": rec.record_id, "value": val})
        return analogs

    def _factor_value(self, dbase: DBase, rec, schema, field_name: str) -> str | None:
        try:
            idx = schema.field_index(field_name)
        except KeyError:
            return None
        indices = rec.sparse_vector.get("indices", [])
        values = rec.sparse_vector.get("values", [])
        if idx not in indices:
            return None
        val = int(values[indices.index(idx)])
        field = schema.field_def(field_name)
        vocab_ref = field.vocabulary_ref or field_name
        return dbase.vocab.reverse_lookup(vocab_ref, val)

    def _read_value(self, dbase: DBase, rec, schema) -> float | None:
        for field_name in ["fold_change", "readout"]:
            try:
                idx = schema.field_index(field_name)
            except KeyError:
                continue
            indices = rec.sparse_vector.get("indices", [])
            values = rec.sparse_vector.get("values", [])
            if idx in indices:
                return float(values[indices.index(idx)])
        return None
