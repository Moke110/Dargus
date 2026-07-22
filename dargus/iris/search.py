"""IrisSearch v0.15.0 — evidence dict API."""

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
            records = dbase.read_shards()
            # Filter records matching this drug and disease
            matching: list[dict] = []
            for rec in records:
                interventions = rec.get("interventions", [])
                primary = next((i for i in interventions if i.get("role") == "primary"), None)
                rec_drug = (primary or {}).get("entity_id", "")
                if rec_drug == drug and rec.get("disease_id") == disease_id:
                    matching.append(rec)

            for endpoint in endpoints:
                endpoint_records = [
                    r
                    for r in matching
                    if r.get("readout_type", "") == endpoint or r.get("endpoint", "") == endpoint
                ]
                if not endpoint_records:
                    continue

                # Read effect values from record dicts
                effects: list[float] = []
                for r in endpoint_records:
                    val = r.get("readout_value")
                    if val is None:
                        val = r.get("fold_change")
                    if val is not None:
                        effects.append(float(val))
                if not effects:
                    continue
                mean_effect = sum(effects) / len(effects)

                # Conservative CI
                lowers: list[float] = []
                uppers: list[float] = []
                for r in endpoint_records:
                    ci95 = r.get("readout_ci95") or {}
                    lo = ci95.get("lower")
                    up = ci95.get("upper")
                    if lo is not None:
                        lowers.append(float(lo))
                    if up is not None:
                        uppers.append(float(up))
                ci_lower = min(lowers) if lowers else mean_effect - 0.5
                ci_upper = max(uppers) if uppers else mean_effect + 0.5

                efficacy_low, efficacy_up = probability_interval_from_effect(
                    mean_effect, ci_lower, ci_upper
                )
                result[drug][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": [r.get("evidence_id", "") for r in endpoint_records],
                    "reasoning_mode": self.name,
                    "confidence_level": "direct_evidence",
                }
        return result
