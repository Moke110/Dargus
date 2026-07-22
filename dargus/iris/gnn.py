from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.base import IrisAgent, PredictionMatrix
from dargus.iris.probability_utils import probability_interval_from_effect
from dargus.reasoning.gnn.graph_builder import DBaseGraphBuilder
from dargus.reasoning.gnn.model import HeteroGnnPredictor


class IrisGnn(IrisAgent):
    name = "Iris-gnn"

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model: HeteroGnnPredictor | None = None

    def predict(
        self,
        dbase: DBase,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        drug_smiles = (embeddings or {}).get("drug_smiles", {})
        builder = DBaseGraphBuilder(dbase)
        graph = builder.build(
            disease_id=disease_id,
            drug_smiles=drug_smiles,
            embedding_dim=self.input_dim,
        )

        has_disease = (
            "disease" in graph.node_id_to_idx and disease_id in graph.node_id_to_idx["disease"]
        )
        if not has_disease:
            return self._empty_result(drug_ids, endpoints)

        if self.model is None:
            num_relations = max(len(graph.edge_types), 1)
            self.model = HeteroGnnPredictor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_relations=num_relations,
            )

        disease_idx_val = graph.node_id_to_idx["disease"][disease_id]
        result: PredictionMatrix = {}
        for drug_id in drug_ids:
            result[drug_id] = {}
            if drug_id not in graph.node_id_to_idx["drug"]:
                for endpoint in endpoints:
                    result[drug_id][endpoint] = self._empty_pred()
                continue

            drug_idx_val = graph.node_id_to_idx["drug"][drug_id]
            pred = self.model.predict_with_ci(
                graph,
                drug_idx=[drug_idx_val],
                disease_idx=[disease_idx_val],
            )
            records = self._supporting_records(dbase, drug_id)
            mean = float(pred["mean"][0])
            ci_lower = float(pred["ci_lower"][0])
            ci_upper = float(pred["ci_upper"][0])
            efficacy_low, efficacy_up = probability_interval_from_effect(mean, ci_lower, ci_upper)
            for endpoint in endpoints:
                result[drug_id][endpoint] = {
                    "efficacy_low": efficacy_low,
                    "efficacy_up": efficacy_up,
                    "supporting_records": records,
                    "reasoning_mode": self.name,
                    "confidence_level": "graph_inference",
                }
        return result

    def _empty_result(self, drug_ids: list[str], endpoints: list[str]) -> PredictionMatrix:
        return {
            drug_id: {endpoint: self._empty_pred() for endpoint in endpoints}
            for drug_id in drug_ids
        }

    def _empty_pred(self) -> dict:
        return {
            "efficacy_low": 0.0,
            "efficacy_up": 1.0,
            "supporting_records": [],
            "reasoning_mode": self.name,
            "confidence_level": "insufficient_data",
        }

    def _supporting_records(self, dbase: DBase, drug_id: str) -> list[str]:
        eids: list[str] = []
        for rec in dbase.read_shards():
            for iv in rec.get("interventions", []):
                if iv.get("entity_id") == drug_id:
                    eid = rec.get("evidence_id", "")
                    if eid:
                        eids.append(eid)
                    break
        return eids[:10]
