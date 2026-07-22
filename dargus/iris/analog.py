"""IrisAnalog v0.15.0 — evidence dict API."""

from __future__ import annotations

from typing import Any

import numpy as np

from dargus.dbase import DBase
from dargus.iris.base import IrisAgent, PredictionMatrix
from dargus.iris.probability_utils import probability_interval_from_effect


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
                        "efficacy_low": 0.0,
                        "efficacy_up": 1.0,
                        "supporting_records": [],
                        "reasoning_mode": self.name,
                        "confidence_level": "insufficient_data",
                    }
                    continue
                values = [r["value"] for r in analog_records]
                mean = float(np.mean(values))
                std = float(np.std(values)) if len(values) > 1 else 1.0
                efficacy_low, efficacy_up = probability_interval_from_effect(
                    mean, mean - 1.96 * std, mean + 1.96 * std
                )
                result[drug_id][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": [r["evidence_id"] for r in analog_records[:10]],
                    "reasoning_mode": self.name,
                    "confidence_level": "analogical_evidence",
                }
        return result

    def _find_analogs(self, dbase: DBase, drug_id: str, disease_id: str) -> list[dict]:
        analogs = []
        for rec in dbase.read_shards():
            interventions = rec.get("interventions", [])
            primary = next((i for i in interventions if i.get("role") == "primary"), None)
            rec_drug = (primary or {}).get("entity_id", "")
            rec_disease = rec.get("disease_id", "")
            if rec_drug == drug_id:
                if rec_disease == disease_id:
                    val = rec.get("readout_value")
                    if val is None:
                        val = rec.get("fold_change")
                    if val is not None:
                        analogs.append(
                            {
                                "evidence_id": rec.get("evidence_id", ""),
                                "value": float(val),
                            }
                        )
        return analogs
