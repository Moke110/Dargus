"""Dargus public API facade.

All adapters (CLI, MCP, Claude Code skill) MUST import only from this module.
No adapter imports anything deeper than ``dargus.api``.
"""

from __future__ import annotations

import logging
from typing import Any

from dargus.dbase import DBase
from dargus.iris.commander import Iris

logger = logging.getLogger(__name__)


def _create_iris_with_lm() -> Iris:
    """Try to create Iris with LifecycleManager injected.

    Attempts to bootstrap a RuntimeContext from the config file and attach
    a LifecycleManager.  Falls back to a plain Iris on any failure (missing
    config, model loading error, etc.) so the API surface never breaks.

    The LifecycleManager is only injected when the runtime is healthy (both
    reasoning LLM and embedding model are present).  A partial bootstrap
    (e.g. missing config) returns a plain Iris to preserve backward compat.
    """
    try:
        from dargus.runtime.bootstrap import bootstrap
        from dargus.runtime.lifecycle import LifecycleManager

        runtime = bootstrap()
        if runtime.healthy:
            lm = LifecycleManager(runtime)
            lm.startup()
            logger.info("API: LifecycleManager attached — using new runtime path")
            return Iris(lifecycle_manager=lm)
        else:
            logger.debug("API: bootstrap produced unhealthy runtime — using direct Iris path")
            return Iris()
    except Exception:
        logger.debug("API: bootstrap failed — falling back to direct Iris path", exc_info=True)
        return Iris()


def predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str],
    max_rounds: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run a full Iris -> Iris multi-round prediction.

    Tries to bootstrap a RuntimeContext and inject a LifecycleManager.
    Falls back to the direct Iris implementation if bootstrap fails.

    Args:
        drug_ids: Drug identifiers to predict for.
        disease_id: Target disease identifier.
        endpoints: Endpoint names (e.g. ``["IC50", "efficacy"]``).
        max_rounds: Maximum Expert dialog rounds (default 5).

    Returns:
        PredictionMatrix: ``{drug_id: {disease_id: {endpoint: {...}}}}``
    """
    iris = _create_iris_with_lm()
    return iris.predict(
        drug_ids=drug_ids,
        disease_id=disease_id,
        endpoints=endpoints,
        max_rounds=max_rounds,
    )


def ingest(datadir: str, reset: bool = False, disease_kb_dir: str | None = None) -> Any:
    """Ingest data into the global D-Base.

    Tries to bootstrap a RuntimeContext and inject a LifecycleManager.
    Falls back to the direct Iris implementation if bootstrap fails.

    Args:
        datadir: Path to directory containing data files.
        reset: If True, clear D-Base before ingestion.
        disease_kb_dir: Optional path to disease knowledge base directory.

    Returns:
        IngestionReport with ``n_records``, ``n_skipped``, ``dbase_size``.
    """
    iris = _create_iris_with_lm()
    return iris.ingest(datadir, disease_kb_dir=disease_kb_dir)


def train(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: use :func:`ingest` instead."""
    import warnings

    warnings.warn("'train' is deprecated, use 'ingest' instead", DeprecationWarning, stacklevel=2)
    return ingest(*args, **kwargs)


def query_dbase(
    disease_id: str | None = None,
    drug_ids: list[str] | None = None,
    levels: list[str] | None = None,
) -> list:
    """Query records from the global D-Base.

    Args:
        disease_id: Filter by disease (optional).
        drug_ids: Filter by drug IDs (optional).
        levels: Filter by biological_level values (optional).

    Returns:
        List of matching D-Base records.
    """
    dbase = DBase.global_instance()
    from dargus.dbase.manager import DBaseManager

    mgr = DBaseManager(dbase)
    return mgr.read_records(
        disease_id=disease_id,
        drug_id=drug_ids[0] if drug_ids and len(drug_ids) == 1 else None,
    )


def status() -> dict[str, Any]:
    """Report global D-Base status.

    Returns:
        Dict with ``dargus_home``, ``n_records``, ``n_templates``.
    """
    iris = Iris()
    return iris.status()


def benchmark(
    strip: dict[str, Any],
    split: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run a bench-full-stack benchmark.

    Tries to bootstrap a RuntimeContext and run through LifecycleManager.
    Falls back to the workflow-level ``run_benchmark`` if bootstrap fails.

    Args:
        strip: Filter dict for extracting matching records from the global D-Base.
        split: Optional split config, e.g. ``{"test_size": 0.2, "random_state": 42}``.
        output_dir: Optional output directory for reports.

    Returns:
        Dict with keys ``metrics``, ``predictions``, ``conditions``.
    """
    task_spec = {
        "workflow": "benchmark",
        "holdout_ids": strip.get("holdout_ids", []),
        "drug_ids": strip.get("drug_ids", []),
        "disease_id": strip.get("disease_id", "unknown"),
        "endpoints": strip.get("endpoints", []),
        "max_rounds": strip.get("max_rounds", 5),
    }
    if split:
        task_spec["split"] = split
    if output_dir:
        task_spec["output_dir"] = output_dir

    # ---- LifecycleManager path ------------------------------------------------
    try:
        from dargus.runtime.bootstrap import bootstrap
        from dargus.runtime.lifecycle import LifecycleManager

        runtime = bootstrap()
        lm = LifecycleManager(runtime)
        lm.startup()
        try:
            result = lm.run_benchmark(task_spec)
            return {
                "metrics": {
                    "accuracy": result.get("accuracy", 0.0),
                    "precision": result.get("precision", 0.0),
                    "recall": result.get("recall", 0.0),
                    "f1": result.get("f1", 0.0),
                },
                "predictions": result.get("report", {}),
                "conditions": strip,
                "n_test": result.get("n_test", 0),
                "status": result.get("status"),
            }
        finally:
            lm.shutdown()
    except Exception:
        logger.debug("API: LifecycleManager benchmark failed — falling back", exc_info=True)

    # ---- Iris benchmark fallback (backward compat) -----------------------------
    try:
        iris = Iris()
        return iris.benchmark(strip=strip, split=split, output_dir=output_dir)
    except NotImplementedError:
        logger.debug("API: Iris.benchmark not implemented — using workflow fallback")
    except Exception:
        logger.debug("API: Iris.benchmark failed — falling back to workflow", exc_info=True)

    # ---- Workflow-level fallback ----------------------------------------------
    from dargus.workflows.benchmark import run_benchmark

    result = run_benchmark(task_spec)
    return {
        "metrics": {
            "accuracy": result.get("accuracy", 0.0),
            "precision": result.get("precision", 0.0),
            "recall": result.get("recall", 0.0),
            "f1": result.get("f1", 0.0),
        },
        "predictions": result.get("report", {}),
        "conditions": strip,
        "n_test": result.get("n_test", 0),
        "status": result.get("status"),
    }


def predict_single_agent(
    agent_name: str,
    drug_ids: list[str],
    disease_id: str,
) -> dict:
    """Run a single Iris-* agent standalone (search, llm, analog, bayes, or gnn).

    Args:
        agent_name: One of ``"iris-search"``, ``"iris-llm"``, ``"iris-analog"``,
                    ``"iris-bayes"``, ``"iris-gnn"``.
        drug_ids: Drug identifiers to predict for.
        disease_id: Target disease identifier.

    Returns:
        PredictionMatrix for the single agent's output.
    """
    from dargus.dbase import DBase
    from dargus.iris.analog import IrisAnalog
    from dargus.iris.bayes import IrisBayes
    from dargus.iris.gnn import IrisGnn
    from dargus.iris.llm import IrisLlm
    from dargus.iris.search import IrisSearch

    name = agent_name.lower()
    mapping: dict[str, Any] = {
        "iris-search": IrisSearch,
        "iris-llm": IrisLlm,
        "iris-analog": IrisAnalog,
        "iris-bayes": IrisBayes,
        "iris-gnn": IrisGnn,
    }
    agent_cls = mapping.get(name)
    if agent_cls is None:
        valid = sorted(mapping.keys())
        raise ValueError(f"Unknown agent: {agent_name!r}. Valid: {valid}")

    dbase = DBase.global_instance()
    agent = agent_cls()
    return agent.predict(dbase, drug_ids or [], disease_id or "", [])


def query_expert(expert_name: str) -> dict:
    """Run a single Expert assessment.

    Note: Full Expert context requires Iris multi-round dialog.
    Individual Expert calls return a stub result.
    """
    return {
        "expert": expert_name,
        "note": "Single Expert assessment — full context requires Iris multi-round dialog",
    }
