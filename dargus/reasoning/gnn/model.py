from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dargus.reasoning.gnn.graph_builder import HeteroGraph


class RGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, relation: int) -> torch.Tensor:
        src, dst = edge_index
        messages = x[src] @ self.weights[relation]
        out = torch.zeros(x.size(0), messages.size(1), device=x.device)
        out.index_add_(0, dst, messages)
        return out + self.bias


class HeteroGnnPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(hidden_dim, hidden_dim, num_relations) for _ in range(num_layers)]
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, graph: HeteroGraph, drug_idx: list[int], disease_idx: list[int]) -> dict:
        h = {k: self.input_proj(v) for k, v in graph.x_dict.items()}

        for conv in self.convs:
            new_h = {k: v.clone() for k, v in h.items()}
            for rel_idx, et in enumerate(graph.edge_types):
                src, rel, dst = et
                edge_index = graph.edge_index_dict[et]
                new_h[dst] = new_h[dst] + conv(
                    h[src], edge_index, relation=rel_idx % self.convs[0].num_relations
                )
            h = {k: F.relu(v) for k, v in new_h.items()}

        drug_emb = h["drug"][drug_idx]
        disease_emb = h["disease"][disease_idx]
        pair = torch.cat([drug_emb, disease_emb], dim=-1)
        out = self.predictor(pair)
        return {"mean": out[:, 0], "log_std": out[:, 1]}

    def predict_with_ci(
        self, graph: HeteroGraph, drug_idx: list[int], disease_idx: list[int]
    ) -> dict:
        self.train()  # keep dropout active for MC uncertainty
        means = []
        stds = []
        with torch.no_grad():
            for _ in range(10):
                out = self.forward(graph, drug_idx, disease_idx)
                means.append(out["mean"])
                stds.append(torch.exp(out["log_std"]))
        mean = torch.stack(means).mean(0)
        total_var = torch.stack(stds).mean(0).pow(2) + torch.stack(means).var(0)
        ci = 1.96 * total_var.sqrt()
        return {
            "mean": mean.numpy(),
            "ci_lower": (mean - ci).numpy(),
            "ci_upper": (mean + ci).numpy(),
        }
