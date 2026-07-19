from __future__ import annotations

from typing import Any

from dargus.iris.base import IrisAgent, PredictionMatrix
from dargus.iris.probability_utils import probability_interval_from_effect


class IrisSearch(IrisAgent):
    """Aggregates direct evidence from D-Base. Most conservative Iris."""

    name = "Iris-search"

    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        result: PredictionMatrix = {drug: {} for drug in drug_ids}
        for drug in drug_ids:
            records = dbase.query(drug_id=drug, disease_id=disease_id)
            for endpoint in endpoints:
                endpoint_records = [
                    r for r in records if self._record_has_endpoint(dbase, r, endpoint)
                ]
                if not endpoint_records:
                    continue
                # Simple average of fold_change
                effects = [self._get_field_value(dbase, r, "fold_change") for r in endpoint_records]
                effects = [e for e in effects if e is not None]
                if not effects:
                    continue
                mean_effect = sum(effects) / len(effects)
                # Conservative CI: max range
                lowers = [self._get_field_value(dbase, r, "ci95_lower") for r in endpoint_records]
                uppers = [self._get_field_value(dbase, r, "ci95_upper") for r in endpoint_records]
                lowers = [lo for lo in lowers if lo is not None]
                uppers = [up for up in uppers if up is not None]
                ci_lower = min(lowers) if lowers else mean_effect - 0.5
                ci_upper = max(uppers) if uppers else mean_effect + 0.5
                efficacy_low, efficacy_up = probability_interval_from_effect(
                    mean_effect, ci_lower, ci_upper
                )
                result[drug][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": [r.record_id for r in endpoint_records],
                    "reasoning_mode": self.name,
                    "confidence_level": "direct_evidence",
                }
        return result

    def _record_has_endpoint(self, dbase: Any, record: Any, endpoint: str) -> bool:
        val = self._get_field_value(dbase, record, "endpoint")
        return val is not None and dbase.vocab.get("endpoint_vocab", endpoint) == int(val)

    def _get_field_value(self, dbase: Any, record: Any, field_name: str) -> float | None:
        schema = dbase._templates.get(record.template_id)
        if schema is None:
            return None
        try:
            idx = schema.field_index(field_name)
        except KeyError:
            return None
        indices = record.sparse_vector.get("indices", [])
        values = record.sparse_vector.get("values", [])
        if idx not in indices:
            return None
        pos = indices.index(idx)
        return values[pos]
