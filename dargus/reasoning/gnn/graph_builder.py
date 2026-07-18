from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dargus.dbase import DBase
from dargus.reasoning.gnn.embedding import disease_onehot_embedding, drug_morgan_embedding


@dataclass
class HeteroGraph:
    node_types: list[str]
    edge_types: list[tuple[str, str, str]]
    x_dict: dict[str, torch.Tensor]
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    node_id_to_idx: dict[str, dict[str, int]] = field(default_factory=dict)


class DBaseGraphBuilder:
    def __init__(self, dbase: DBase):
        self.dbase = dbase

    def build(
        self,
        disease_id: str | None = None,
        drug_smiles: dict[str, str] | None = None,
        embedding_dim: int = 128,
    ) -> HeteroGraph:
        records = self.dbase.list_records()

        drug_ids: set[str] = set()
        target_ids: set[str] = set()
        disease_ids: set[str] = set()
        assay_ids: set[str] = set()

        edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {
            ("drug", "interacts", "target"): [],
            ("drug", "tested_in", "assay"): [],
            ("assay", "relates_to", "disease"): [],
            ("disease", "similar_to", "disease"): [],
        }

        drug_to_idx: dict[str, int] = {}
        target_to_idx: dict[str, int] = {}
        disease_to_idx: dict[str, int] = {}
        assay_to_idx: dict[str, int] = {}

        def _get_idx(mapping: dict[str, int], key: str) -> int:
            if key not in mapping:
                mapping[key] = len(mapping)
            return mapping[key]

        for rec in records:
            schema = self.dbase._templates.get(rec.template_id)
            if schema is None:
                continue

            drug = self._factor_value(rec, schema, "drug_id")
            target = self._factor_value(rec, schema, "target_id")
            disease = self._factor_value(rec, schema, "disease_id")

            if drug:
                drug_ids.add(drug)
            if target:
                target_ids.add(target)
            if disease:
                disease_ids.add(disease)

            assay_key = f"{rec.template_id}:{rec.record_id}"
            assay_ids.add(assay_key)
            assay_idx = _get_idx(assay_to_idx, assay_key)

            if drug:
                drug_idx = _get_idx(drug_to_idx, drug)
                edges[("drug", "tested_in", "assay")].append((drug_idx, assay_idx))
            if drug and target:
                drug_idx = _get_idx(drug_to_idx, drug)
                target_idx = _get_idx(target_to_idx, target)
                edges[("drug", "interacts", "target")].append((drug_idx, target_idx))
            if disease:
                disease_idx = _get_idx(disease_to_idx, disease)
                edges[("assay", "relates_to", "disease")].append((assay_idx, disease_idx))

        # Add disease similarity edges (dummy: fully connected small set)
        disease_list = list(disease_to_idx.keys())
        for i, d1 in enumerate(disease_list):
            for j, d2 in enumerate(disease_list):
                if i != j:
                    edges[("disease", "similar_to", "disease")].append(
                        (disease_to_idx[d1], disease_to_idx[d2])
                    )

        # Node features
        drug_x = []
        for drug in drug_to_idx:
            smiles = (drug_smiles or {}).get(drug)
            if smiles:
                feat = drug_morgan_embedding(smiles)[:embedding_dim]
            else:
                feat = np.zeros(embedding_dim, dtype=float)
            drug_x.append(feat)

        target_x = [np.zeros(embedding_dim, dtype=float) for _ in target_to_idx]
        disease_x = [disease_onehot_embedding(d, dim=embedding_dim) for d in disease_to_idx]
        assay_x = [np.zeros(embedding_dim, dtype=float) for _ in assay_to_idx]

        x_dict = {
            "drug": torch.tensor(np.array(drug_x), dtype=torch.float),
            "target": torch.tensor(np.array(target_x), dtype=torch.float),
            "disease": torch.tensor(np.array(disease_x), dtype=torch.float),
            "assay": torch.tensor(np.array(assay_x), dtype=torch.float),
        }

        edge_index_dict = {
            et: torch.tensor(np.array(pairs).T, dtype=torch.long)
            for et, pairs in edges.items()
            if pairs
        }

        return HeteroGraph(
            node_types=list(x_dict.keys()),
            edge_types=list(edge_index_dict.keys()),
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            node_id_to_idx={
                "drug": drug_to_idx,
                "target": target_to_idx,
                "disease": disease_to_idx,
                "assay": assay_to_idx,
            },
        )

    def _factor_value(
        self,
        rec: Any,
        schema: Any,
        field_name: str,
    ) -> str | None:
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
        return self.dbase.vocab.reverse_lookup(vocab_ref, val)
