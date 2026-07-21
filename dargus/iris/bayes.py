from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.base import IrisAgent, PredictionMatrix
from dargus.iris.probability_utils import probability_interval_from_effect


class IrisBayes(IrisAgent):
    name = "Iris-bayes"

    def __init__(self, draws: int = 300, tune: int = 300, chains: int = 2):
        self.draws = draws
        self.tune = tune
        self.chains = chains

    def predict(
        self,
        dbase: DBase,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        from dargus.reasoning.bayesian.model import HierarchicalBayesianModel
        from dargus.reasoning.bayesian.records_adapter import LEVEL_ORDER, RecordsAdapter

        adapter = RecordsAdapter(dbase)
        result: PredictionMatrix = {}
        for drug_id in drug_ids:
            result[drug_id] = {}
            for endpoint in endpoints:
                y, level_idx, group_idx, records = adapter.to_arrays(
                    drug_ids=[drug_id],
                    disease_id=disease_id,
                    endpoints=[endpoint],
                )
                if len(y) == 0:
                    result[drug_id][endpoint] = {
                        "efficacy_low": 0.0,
                        "efficacy_up": 1.0,
                        "supporting_records": [],
                        "reasoning_mode": self.name,
                        "confidence_level": "insufficient_data",
                    }
                    continue

                model = HierarchicalBayesianModel(
                    y, level_idx, group_idx, n_levels=len(LEVEL_ORDER)
                )
                model.fit(draws=self.draws, tune=self.tune, chains=self.chains)
                pred = model.predict(level=LEVEL_ORDER.index("clinical"), group=0)
                efficacy_low, efficacy_up = probability_interval_from_effect(
                    pred["mean"], pred["ci_lower"], pred["ci_upper"]
                )
                result[drug_id][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": records[:10],
                    "reasoning_mode": self.name,
                    "confidence_level": "multi_level_evidence" if len(y) > 1 else "single_study",
                }
        return result
