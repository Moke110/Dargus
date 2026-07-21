"""Dargus public API facade.

All adapters (CLI, MCP, Claude Code skill) MUST import only from this module.
No adapter imports anything deeper than ``dargus.api``.
"""

from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.commander import Iris


def predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str],
    max_rounds: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run a full Iris -> IrisExpert multi-round prediction.

    Args:
        drug_ids: Drug identifiers to predict for.
        disease_id: Target disease identifier.
        endpoints: Endpoint names (e.g. ``["IC50", "efficacy"]``).
        max_rounds: Maximum Expert dialog rounds (default 5).

    Returns:
        PredictionMatrix: ``{drug_id: {disease_id: {endpoint: {...}}}}``
    """
    iris = Iris()
    return iris.predict(
        drug_ids=drug_ids,
        disease_id=disease_id,
        endpoints=endpoints,
        max_rounds=max_rounds,
    )


def train(datadir: str, reset: bool = False) -> Any:
    """Ingest data into the global D-Base.

    Args:
        datadir: Path to directory containing data files.
        reset: If True, clear D-Base before ingestion.

    Returns:
        TrainingReport with ``n_records``, ``n_skipped``, ``dbase_size``.
    """
    iris = Iris()
    return iris.train(datadir)


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
    return dbase.query(
        disease_id=disease_id,
        drug_id=drug_ids[0] if drug_ids and len(drug_ids) == 1 else None,
        template_id=None,
    )


def status() -> dict[str, Any]:
    """Report global D-Base status.

    Returns:
        Dict with ``dargus_home``, ``n_records``, ``n_templates``.
    """
    iris = Iris()
    return iris.status()


def benchmark(
    config_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a benchmark configuration.

    Args:
        config_path: Path to a YAML benchmark config file.
        config: Inline benchmark config dict (alternative to config_path).

    Returns:
        Dict with benchmark metrics.
    """
    iris = Iris()
    if config_path is not None:
        return iris.benchmark(config_path)
    raise NotImplementedError("inline config dict not yet supported — use config_path")


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

    Note: Full Expert context requires IrisExpert multi-round dialog.
    Individual Expert calls return a stub result.
    """
    return {
        "expert": expert_name,
        "note": "Single Expert assessment — full context requires IrisExpert multi-round dialog",
    }
