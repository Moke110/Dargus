"""IrisGnn — graph neural network efficacy prediction, wrapped as Tool."""

from __future__ import annotations

from typing import Any

from dargus.dbase import DBase

try:
    from dargus.iris.gnn import IrisGnn

    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False


def gnn_predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
) -> dict[str, Any]:
    if not _GNN_AVAILABLE:
        raise RuntimeError("IrisGnn requires PyTorch (not installed)")
    dbase = DBase.global_instance()
    agent = IrisGnn()
    return agent.predict(dbase, drug_ids, disease_id, endpoints or [])
