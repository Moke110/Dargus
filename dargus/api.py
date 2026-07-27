"""Dargus public API facade.

All adapters (CLI, Claude Code skill) MUST import only from this module.
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
    if reset:
        from dargus.dbase import DBase
        from dargus.dbase.manager import DBaseManager

        dbase = DBase.global_instance()
        manager = DBaseManager(dbase)
        manager.reset()
        logger.info("API: D-Base reset before ingestion")

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


# ---------------------------------------------------------------------------
# Session / Environment
# ---------------------------------------------------------------------------


def init() -> None:
    """Initialize Dargus session — load .env and configure logging."""
    from dargus._env import load_dotenv

    load_dotenv()


def has_api_key() -> bool:
    """Check if DARGUS_LLM_API_KEY is set in the environment."""
    import os

    return bool(os.environ.get("DARGUS_LLM_API_KEY"))


# ---------------------------------------------------------------------------
# Natural Language Interface
# ---------------------------------------------------------------------------


def ask(query: str) -> str:
    """Route a natural-language query through Iris.

    Args:
        query: Natural language query string.

    Returns:
        Iris response as a human-readable string.
    """
    iris = _create_iris_with_lm()
    return iris.process_query(query)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_llm_config() -> dict[str, Any]:
    """Get current LLM configuration.

    Returns:
        Dict with keys: provider, model, base_url, temperature, max_tokens, has_api_key.
    """
    from dargus.config.paths import get_config_path

    config_path = get_config_path()
    if not config_path.exists():
        return {
            "provider": "openai_compatible",
            "model": "",
            "base_url": "",
            "temperature": 0.0,
            "max_tokens": 2048,
            "has_api_key": has_api_key(),
        }

    import yaml

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    llm_cfg = cfg.get("llm", {})
    return {
        "provider": llm_cfg.get("provider", "openai_compatible"),
        "model": llm_cfg.get("model", ""),
        "base_url": llm_cfg.get("base_url", ""),
        "temperature": llm_cfg.get("temperature", 0.0),
        "max_tokens": llm_cfg.get("max_tokens", 2048),
        "has_api_key": has_api_key(),
    }


def save_llm_config(model: str, base_url: str, provider: str = "openai_compatible") -> None:
    """Save LLM configuration to config file.

    Args:
        model: Model name.
        base_url: API base URL.
        provider: Provider name (default: openai_compatible).
    """
    from dargus.config.paths import get_config_path

    config_path = get_config_path()

    # Load existing config
    cfg: dict = {}
    if config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    # Update llm section
    cfg.setdefault("llm", {})
    cfg["llm"]["model"] = model
    cfg["llm"]["base_url"] = base_url
    cfg["llm"]["provider"] = provider

    # Write back
    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)


def set_api_key(provider: str, key: str) -> str:
    """Write API key to .env file.

    Args:
        provider: Provider name (currently unused, for future per-provider keys).
        key: API key.

    Returns:
        Path to the .env file.
    """
    from dargus._env import write_dotenv

    env_path = write_dotenv("DARGUS_LLM_API_KEY", key)
    return str(env_path)


def test_llm_connection(model: str, base_url: str, api_key: str | None = None) -> dict[str, Any]:
    """Test LLM connection.

    Args:
        model: Model name.
        base_url: API base URL.
        api_key: Optional API key (uses env var if not provided).

    Returns:
        Dict with keys: ok, model, latency_ms, error.
    """
    import os

    from dargus.models.compat import DargusLLM, check_llm_connection

    key = api_key or os.environ.get("DARGUS_LLM_API_KEY")
    llm = DargusLLM(model=model, base_url=base_url, api_key=key)
    return check_llm_connection(llm)


# ---------------------------------------------------------------------------
# D-Base Administration
# ---------------------------------------------------------------------------


def generate_clear_dbase_code() -> str:
    """Generate a confirmation code for clearing the D-Base.

    Returns:
        A random hex confirmation code (10 characters).
    """
    import secrets

    return secrets.token_hex(5)


def clear_dbase(confirm_code: str, expected_code: str) -> bool:
    """Clear all records from the global D-Base.

    This is a destructive operation that requires two-step confirmation:
    1. Call generate_clear_dbase_code() to get a code
    2. Display the code to the user and get their input
    3. Call this function with both the user's input and the expected code

    Args:
        confirm_code: The confirmation code provided by the user.
        expected_code: The expected confirmation code from generate_clear_dbase_code().

    Returns:
        True if successful, False otherwise (including code mismatch).
    """
    import secrets

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    # Verify confirmation code
    if not secrets.compare_digest(confirm_code, expected_code):
        logger.warning("API: clear_dbase called with mismatched confirmation code")
        return False

    try:
        dbase = DBase.global_instance()
        manager = DBaseManager(dbase)
        manager.reset()
        logger.info("API: D-Base cleared")
        return True
    except Exception:
        logger.exception("API: Failed to clear D-Base")
        return False


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


def list_test_modules() -> list[dict[str, Any]]:
    """List available test modules.

    Returns:
        List of dicts with keys: name, n_tests.
    """
    from pathlib import Path

    test_dir = Path(__file__).resolve().parent / "tests"
    modules = []

    for p in sorted(test_dir.iterdir()):
        if p.is_dir() and (p / "__init__.py").exists():
            n_tests = len(list(p.glob("test_*.py")))
            modules.append({"name": p.name, "n_tests": n_tests})

    return modules


def run_tests(module: str | None = None) -> int:
    """Run pytest on test suite.

    Args:
        module: Optional module name to run tests for (None = all tests).

    Returns:
        Pytest exit code.
    """
    from pathlib import Path

    import pytest

    test_dir = str(Path(__file__).resolve().parent / "tests")
    if module:
        test_dir = str(Path(test_dir) / module)

    return pytest.main(["-q", test_dir])


def test_write_evidence(raw: dict, source_id: str = "test-dbase:cli") -> dict[str, Any]:
    """Write a single evidence record to test D-Base.

    Args:
        raw: Raw evidence data dict.
        source_id: Source identifier for metadata.

    Returns:
        Dict with keys: evidence_id, biological_level, readout_type, status.
    """
    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)

    evidence = manager.build_evidence(
        raw,
        source_metadata={"type": "file_path", "id": source_id},
    )
    wrote = manager.write_record(evidence)

    return {
        "evidence_id": evidence["evidence_id"],
        "biological_level": evidence.get("biological_level", "?"),
        "readout_type": evidence.get("readout_type", "?"),
        "status": "added" if wrote is True else "duplicate — skipped",
    }


def test_bulk_input(directory: str) -> dict[str, Any]:
    """Bulk-write evidence .json files from directory to test D-Base.

    Args:
        directory: Directory containing .json evidence files.

    Returns:
        Dict with keys: directory, total, added, duplicates, hard_rejects, errors, elapsed.
    """
    import json
    import time
    from pathlib import Path

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        return {
            "directory": str(dir_path),
            "total": 0,
            "added": 0,
            "duplicates": 0,
            "hard_rejects": 0,
            "errors": 1,
            "elapsed": 0.0,
            "error_details": [f"Directory not found: {dir_path}"],
        }

    json_files = sorted(dir_path.glob("*.json"))
    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)

    added = 0
    duplicates = 0
    hard_rejects = 0
    errors = 0
    error_details: list[str] = []
    t0 = time.perf_counter()

    for jf in json_files:
        try:
            raw_data = json.loads(jf.read_text(encoding="utf-8"))
            evidence = manager.build_evidence(
                raw_data,
                source_metadata={"type": "file_path", "id": f"test-bulk:{jf.name}"},
            )
            result = manager.write_record(evidence)
            if result is True:
                added += 1
            else:
                duplicates += 1
        except json.JSONDecodeError as exc:
            hard_rejects += 1
            error_details.append(f"{jf.name}: JSON parse error — {exc}")
        except ValueError as exc:
            hard_rejects += 1
            error_details.append(f"{jf.name}: {str(exc)[:120]}")
        except Exception as exc:
            errors += 1
            error_details.append(f"{jf.name}: {exc}")

    elapsed = time.perf_counter() - t0

    return {
        "directory": str(dir_path),
        "total": len(json_files),
        "added": added,
        "duplicates": duplicates,
        "hard_rejects": hard_rejects,
        "errors": errors,
        "elapsed": elapsed,
        "error_details": error_details,
    }


def test_ingest_dir(directory: str) -> dict[str, Any]:
    """Run ingest workflow on test data directory.

    Args:
        directory: Directory containing test data files.

    Returns:
        Dict with keys: directory, total, added, duplicates, hard_rejects, errors, elapsed.
    """
    import time
    from pathlib import Path

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        return {
            "directory": str(dir_path),
            "total": 0,
            "added": 0,
            "duplicates": 0,
            "hard_rejects": 0,
            "errors": 1,
            "elapsed": 0.0,
            "error_details": [f"Directory not found: {dir_path}"],
        }

    # Scan for converters
    from dargus.ingestion.converters.gdsc import GdscConverter
    from dargus.ingestion.converters.tdc_admet import TdcAdmetConverter
    from dargus.ingestion.converters.tdc_dti import TdcDtiConverter
    from dargus.ingestion.converters.top_clinical import TopClinicalConverter

    file_map: list[tuple] = []
    for fp in sorted(dir_path.iterdir()):
        if not fp.is_file():
            continue
        name = fp.name.lower()

        if "gdsc" in name and name.endswith(".csv"):
            file_map.append((GdscConverter(), fp))
        elif any(kw in name for kw in ("bindingdb", "davis", "kiba")):
            assay = "affinity"
            if "ic50" in name:
                assay = "IC50"
            elif "ki" in name:
                assay = "Ki"
            elif "kd" in name:
                assay = "Kd"
            file_map.append((TdcDtiConverter(assay), fp))
        elif "solubility" in name:
            file_map.append((TdcAdmetConverter("solubility"), fp))
        elif "bioavailability" in name:
            file_map.append((TdcAdmetConverter("bioavailability"), fp))
        elif "cyp3a4" in name:
            file_map.append((TdcAdmetConverter("cyp3a4_substrate"), fp))
        elif any(
            kw in name
            for kw in (
                "cyp",
                "caco2",
                "bbb",
                "half_life",
                "clearance",
                "vdss",
                "ppbr",
                "ames",
                "carcinogens",
                "ld50",
            )
        ):
            file_map.append((TdcAdmetConverter("admet"), fp))
        elif "top" in name and "clinical" in name:
            file_map.append((TopClinicalConverter(), fp))

    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)

    added = 0
    duplicates = 0
    hard_rejects = 0
    errors = 0
    error_details: list[str] = []
    t0 = time.perf_counter()

    for conv_instance, fp in file_map:
        try:
            raw_rows = conv_instance.convert(fp)
            for row_idx, raw in enumerate(raw_rows):
                try:
                    evidence = manager.build_evidence(
                        raw,
                        source_metadata={
                            "type": "file_path",
                            "id": f"test-ingest:{fp.name}:{row_idx}",
                        },
                    )
                    result = manager.write_record(evidence)
                    if result is True:
                        added += 1
                    else:
                        duplicates += 1
                except ValueError as exc:
                    hard_rejects += 1
                    error_details.append(f"{fp.name} row {row_idx}: {str(exc)[:120]}")
                except Exception as exc:
                    errors += 1
                    error_details.append(f"{fp.name} row {row_idx}: {exc}")
        except Exception as exc:
            errors += 1
            error_details.append(f"{fp.name}: convert failed — {exc}")

    elapsed = time.perf_counter() - t0

    # Rebuild view (best-effort)
    try:
        dbase.rebuild_view()
    except Exception:
        pass

    return {
        "directory": str(dir_path),
        "total": len(file_map),
        "added": added,
        "duplicates": duplicates,
        "hard_rejects": hard_rejects,
        "errors": errors,
        "elapsed": elapsed,
        "error_details": error_details,
    }


def get_test_config(key: str, default: str = "") -> str:
    """Get a test configuration value.

    Args:
        key: Config key (e.g., 'bulk_input_dir', 'ingest_dir').
        default: Default value if key not found.

    Returns:
        Config value or default.
    """
    from dargus.config.paths import get_config_path

    config_path = get_config_path()
    if not config_path.exists():
        return default

    import yaml

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    return cfg.get("test", {}).get(key, default)


def set_test_config(key: str, value: str) -> None:
    """Set a test configuration value.

    Args:
        key: Config key (e.g., 'bulk_input_dir', 'ingest_dir').
        value: Config value.
    """
    from dargus.config.paths import get_config_path

    config_path = get_config_path()

    # Load existing config
    cfg: dict = {}
    if config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    # Update test section
    cfg.setdefault("test", {})[key] = value

    # Write back
    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)


def write_ingest_report(result: dict[str, Any]) -> str:
    """Write ingest test report to markdown file.

    Args:
        result: Result dict from test_ingest_dir.

    Returns:
        Path to the written report file.
    """
    from datetime import datetime
    from pathlib import Path

    report_path = Path(result["directory"]).parent / "Ingest-test-report.md"

    lines = [
        "# Ingest Test Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data directory**: `{result['directory']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Files processed | {result['total']} |",
        f"| Evidence records added | {result['added']} |",
        f"| Duplicates skipped | {result['duplicates']} |",
        f"| Hard rejects | {result['hard_rejects']} |",
        f"| Errors | {result['errors']} |",
        f"| Total time | {result['elapsed']:.1f}s |",
        "",
    ]

    if result.get("error_details"):
        lines.append("## Error Details")
        lines.append("")
        for detail in result["error_details"][:20]:
            lines.append(f"- {detail}")
        if len(result["error_details"]) > 20:
            lines.append(f"- ... and {len(result['error_details']) - 20} more")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
