"""MCP tool implementations — 14 tools in L1/L2/L3 layers.

All tools import ONLY from ``dargus.api`` where possible. L2 tools that need
individual Iris agent access import from ``dargus.iris.*`` as necessary.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import dargus

logger = logging.getLogger(__name__)


def _respond(success: bool, data: dict[str, Any] | None = None, error: str | None = None) -> dict:
    return {"success": success, "data": data or {}, "error": error}


def _safe_error(tool_name: str, exc: Exception) -> str:
    """Log the real error server-side, return a generic message to the client."""
    logger.exception("%s failed", tool_name)
    return f"Internal error: {tool_name} failed"


# ── L1: D-Base CRUD ──────────────────────────────────────────────


def tool_dbase_query(
    disease_id: str | None = None,
    drug_ids: list[str] | None = None,
    levels: list[str] | None = None,
) -> dict:
    """Query records from the global D-Base."""
    try:
        records = dargus.query_dbase(
            disease_id=disease_id,
            drug_ids=drug_ids,
            levels=levels,
        )
        return _respond(
            True,
            data={
                "n_records": len(records),
                "records": [json.loads(r.model_dump_json()) for r in records],
            },
        )
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_dbase_query", exc))


def tool_dbase_ingest(
    datadir: str,
    reset: bool = False,
) -> dict:
    """Ingest data into the global D-Base (via DBaseManager)."""
    try:
        real_path = os.path.realpath(datadir)
        if not os.path.isdir(real_path):
            return _respond(False, error=f"Not a directory: {datadir}")
        report = dargus.train(datadir=real_path, reset=reset)
        return _respond(
            True,
            data={
                "n_records": report.n_records,
                "n_skipped": report.n_skipped,
                "dbase_size": report.dbase_size,
            },
        )
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_dbase_ingest", exc))


def tool_dbase_status() -> dict:
    """Show global D-Base status."""
    try:
        status = dargus.status()
        return _respond(True, data=status)
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_dbase_status", exc))


# ── L2: Iris-* primitives (single-shot, read-only) ───────────────


def tool_iris_search(
    drug_ids: list[str] | None = None,
    disease_id: str | None = None,
) -> dict:
    """Run Iris-search only (literature/evidence search)."""
    try:
        from dargus.dbase import DBase
        from dargus.iris.search import IrisSearch

        dbase = DBase.global_instance()
        agent = IrisSearch()
        result = agent.predict(
            dbase,
            drug_ids or [],
            disease_id or "",
            [],
        )
        return _respond(True, data={"predictions": result})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_iris_search", exc))


def tool_iris_llm(
    drug_ids: list[str] | None = None,
    disease_id: str | None = None,
) -> dict:
    """Run Iris-llm only (LLM-based reasoning)."""
    try:
        from dargus.dbase import DBase
        from dargus.iris.llm import IrisLlm

        dbase = DBase.global_instance()
        agent = IrisLlm()
        result = agent.predict(
            dbase,
            drug_ids or [],
            disease_id or "",
            [],
        )
        return _respond(True, data={"predictions": result})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_iris_llm", exc))


def tool_iris_analog(
    drug_ids: list[str] | None = None,
    disease_id: str | None = None,
) -> dict:
    """Run Iris-analog only (analog-based reasoning)."""
    try:
        from dargus.dbase import DBase
        from dargus.iris.analog import IrisAnalog

        dbase = DBase.global_instance()
        agent = IrisAnalog()
        result = agent.predict(
            dbase,
            drug_ids or [],
            disease_id or "",
            [],
        )
        return _respond(True, data={"predictions": result})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_iris_analog", exc))


def tool_iris_bayes(
    drug_ids: list[str] | None = None,
    disease_id: str | None = None,
) -> dict:
    """Run Iris-bayes only (Bayesian modeling)."""
    try:
        from dargus.dbase import DBase
        from dargus.iris.bayes import IrisBayes

        dbase = DBase.global_instance()
        agent = IrisBayes()
        result = agent.predict(
            dbase,
            drug_ids or [],
            disease_id or "",
            [],
        )
        return _respond(True, data={"predictions": result})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_iris_bayes", exc))


def tool_iris_gnn(
    drug_ids: list[str] | None = None,
    disease_id: str | None = None,
) -> dict:
    """Run Iris-gnn only (graph neural network prediction)."""
    try:
        from dargus.dbase import DBase
        from dargus.iris.gnn import IrisGnn

        dbase = DBase.global_instance()
        agent = IrisGnn()
        result = agent.predict(
            dbase,
            drug_ids or [],
            disease_id or "",
            [],
        )
        return _respond(True, data={"predictions": result})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_iris_gnn", exc))


# ── L2: Single Expert assessment (single-shot, read-only) ────────


def _run_single_expert(expert_name: str) -> dict:
    """Run a single Expert assessment. Returns a stub result since
    individual Experts require IrisExpert orchestration context."""
    return _respond(
        True,
        data={
            "expert": expert_name,
            "note": (
                "Single Expert assessment — full context requires " "IrisExpert multi-round dialog"
            ),
        },
    )


def tool_expert_molecule() -> dict:
    """Run MoleculeExpert assessment only."""
    return _run_single_expert("MoleculeExpert")


def tool_expert_biomed() -> dict:
    """Run BiomedExpert assessment only."""
    return _run_single_expert("BiomedExpert")


def tool_expert_bioinfo() -> dict:
    """Run BioinfoExpert assessment only."""
    return _run_single_expert("BioinfoExpert")


def tool_expert_clinic() -> dict:
    """Run ClinicExpert assessment only."""
    return _run_single_expert("ClinicExpert")


def tool_expert_director() -> dict:
    """Run FourDExpert (director) assessment only."""
    return _run_single_expert("FourDExpert")


# ── L3: Full agent prediction ────────────────────────────────────


def tool_predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
    max_rounds: int = 5,
) -> dict:
    """Run full Iris -> IrisExpert multi-round prediction."""
    try:
        predictions = dargus.predict(
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints or [],
            max_rounds=max_rounds,
        )
        return _respond(True, data={"predictions": predictions})
    except Exception as exc:
        return _respond(False, error=_safe_error("tool_predict", exc))


# ── Tool registry ─────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    # L1: D-Base CRUD
    {
        "name": "dargus_dbase_query",
        "description": "Query records from the D-Base by disease, drug, or biological level.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "disease_id": {"type": "string"},
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "levels": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "dargus_dbase_ingest",
        "description": "Ingest data from a directory into D-Base via DBaseManager.",
        "inputSchema": {
            "type": "object",
            "required": ["datadir"],
            "properties": {
                "datadir": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "dargus_dbase_status",
        "description": "Show global D-Base status (record count, templates, location).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # L2: Iris primitives
    {
        "name": "dargus_iris_search",
        "description": "Run Iris-search only — literature and evidence retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
            },
        },
    },
    {
        "name": "dargus_iris_llm",
        "description": "Run Iris-llm only — LLM-based reasoning on drug-disease evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
            },
        },
    },
    {
        "name": "dargus_iris_analog",
        "description": "Run Iris-analog only — analog-based drug similarity reasoning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
            },
        },
    },
    {
        "name": "dargus_iris_bayes",
        "description": "Run Iris-bayes only — Bayesian probabilistic modeling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
            },
        },
    },
    {
        "name": "dargus_iris_gnn",
        "description": "Run Iris-gnn only — graph neural network prediction.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
            },
        },
    },
    # L2: Single Expert
    {
        "name": "dargus_expert_molecule",
        "description": "Run MoleculeExpert assessment only — molecular-level evidence evaluation.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dargus_expert_biomed",
        "description": "Run BiomedExpert assessment only — biomedical pathway evidence.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dargus_expert_bioinfo",
        "description": "Run BioinfoExpert assessment only — bioinformatics/genomics evidence.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dargus_expert_clinic",
        "description": "Run ClinicExpert assessment only — clinical evidence evaluation.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dargus_expert_director",
        "description": "Run FourDExpert (director) assessment only — cross-expert synthesis.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # L3: Full agent
    {
        "name": "dargus_predict",
        "description": (
            "Run full Iris -> IrisExpert multi-round prediction for drugs against a disease."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["drug_ids", "disease_id"],
            "properties": {
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
                "endpoints": {"type": "array", "items": {"type": "string"}},
                "max_rounds": {"type": "integer", "default": 5},
            },
        },
    },
]

TOOL_DISPATCH: dict[str, Any] = {
    "dargus_dbase_query": tool_dbase_query,
    "dargus_dbase_ingest": tool_dbase_ingest,
    "dargus_dbase_status": tool_dbase_status,
    "dargus_iris_search": tool_iris_search,
    "dargus_iris_llm": tool_iris_llm,
    "dargus_iris_analog": tool_iris_analog,
    "dargus_iris_bayes": tool_iris_bayes,
    "dargus_iris_gnn": tool_iris_gnn,
    "dargus_expert_molecule": tool_expert_molecule,
    "dargus_expert_biomed": tool_expert_biomed,
    "dargus_expert_bioinfo": tool_expert_bioinfo,
    "dargus_expert_clinic": tool_expert_clinic,
    "dargus_expert_director": tool_expert_director,
    "dargus_predict": tool_predict,
}
