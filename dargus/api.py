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
    """Create Iris through the runtime's AgentFactory.

    Bootstraps a DargusRuntime from the config file. Per design/2_runtime_structure.md
    the runtime starts healthy and entry points refuse new sessions while
    unhealthy — there is no silent fallback to a runtime-less path. On an
    unrecoverable bootstrap failure the runtime is marked unhealthy and the
    entry point raises.
    """
    from dargus.runtime.bootstrap import bootstrap

    try:
        runtime = bootstrap()
    except Exception as exc:
        logger.error("API: runtime bootstrap failed: %s", exc)
        raise RuntimeError(f"DargusRuntime failed to start: {exc} — refusing new session") from exc
    runtime.ensure_healthy()
    return runtime.agent_factory.iris()


def predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str],
    max_rounds: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run a full Iris -> Iris multi-round prediction.

    Bootstraps the DargusRuntime and creates Iris via the AgentFactory;
    refuses the session when the runtime is unhealthy.

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


def ingest(datadir: str, reset: bool = False, disease_kb_dir: str | None = None) -> dict[str, Any]:
    """Ingest data into the global D-Base.

    Bootstraps the DargusRuntime and creates Iris via the AgentFactory;
    refuses the session when the runtime is unhealthy.

    Args:
        datadir: Path to directory containing data files.
        reset: If True, clear D-Base before ingestion.
        disease_kb_dir: Optional path to disease knowledge base directory.

    Returns:
        IngestResult dict with ``n_records``, ``n_duplicates``, ``n_errors``, etc.
    """
    if reset:
        from dargus.dbase import DBase
        from dargus.dbase.store import DBaseStore

        dbase = DBase.global_instance()
        manager = DBaseStore(dbase)
        manager.reset()
        logger.info("API: D-Base reset before ingestion")

    iris = _create_iris_with_lm()
    return iris.ingest(datadir, disease_kb_dir=disease_kb_dir)


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
    from dargus.dbase.store import DBaseStore

    mgr = DBaseStore(dbase)
    return mgr.read_records(
        disease_id=disease_id,
        x_entity=drug_ids[0] if drug_ids and len(drug_ids) == 1 else None,
    )


def status() -> dict[str, Any]:
    """Report global D-Base status.

    Returns:
        Dict with ``dargus_home``, ``n_records``, ``n_templates``.
    """
    iris = _create_iris_with_lm()
    return iris.status()


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
            "provider": "openai",
            "model": "",
            "base_url": "https://api.deepseek.com/v1",
            "temperature": 0.0,
            "max_tokens": 2048,
            "has_api_key": has_api_key(),
        }

    import yaml

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # v1.0.0: read from models.reasoning block (single source of truth)
    reasoning_cfg = (cfg.get("models") or {}).get("reasoning", {})
    return {
        "provider": reasoning_cfg.get("provider", "deepseek"),
        "model": reasoning_cfg.get("model", ""),
        "base_url": reasoning_cfg.get("base_url", "https://api.deepseek.com"),
        "temperature": reasoning_cfg.get("temperature", 0.0),
        "max_tokens": reasoning_cfg.get("max_tokens", 2048),
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

    # v1.0.0: write into models.reasoning block (single source of truth)
    cfg.setdefault("models", {}).setdefault("reasoning", {})
    cfg["models"]["reasoning"]["model"] = model
    cfg["models"]["reasoning"]["base_url"] = base_url
    cfg["models"]["reasoning"]["provider"] = provider

    # Clean up legacy llm block if present
    cfg.pop("llm", None)

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
    """Test LLM connection against an OpenAI-compatible endpoint.

    Args:
        model: Model name.
        base_url: API base URL.
        api_key: Optional API key (uses env var if not provided).

    Returns:
        Dict with keys: ok, model, latency_ms, error.
    """
    import os
    import time

    import httpx

    key = api_key or os.environ.get("DARGUS_LLM_API_KEY")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with just: OK"}],
        "temperature": 0.0,
        "max_tokens": 8,
    }

    t0 = time.monotonic()
    try:
        response = httpx.post(url, headers=headers, json=body, timeout=30.0)
        response.raise_for_status()
        response.json()["choices"][0]["message"]["content"]
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {"ok": True, "model": model, "latency_ms": latency_ms}
    except Exception as exc:
        return {"ok": False, "model": model, "error": str(exc)}


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
    from dargus.dbase.store import DBaseStore

    # Verify confirmation code
    if not secrets.compare_digest(confirm_code, expected_code):
        logger.warning("API: clear_dbase called with mismatched confirmation code")
        return False

    try:
        dbase = DBase.global_instance()
        manager = DBaseStore(dbase)
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
        Dict with keys: evidence_id, biological_level, y_type, status.
    """
    from dargus.dbase import DBase
    from dargus.dbase.store import DBaseStore

    dbase = DBase.global_instance()
    manager = DBaseStore(dbase)

    evidence = manager.build_evidence(
        raw,
        source_metadata={"type": "file_path", "id": source_id},
    )
    wrote = manager.write_record(evidence)

    return {
        "evidence_id": evidence["evidence_id"],
        "biological_level": evidence.get("biological_level", "?"),
        "y_type": (evidence.get("y") or {}).get("type", "?"),
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
    from dargus.dbase.store import DBaseStore

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
    manager = DBaseStore(dbase)

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

    # Ingest via the single task_spec calling convention
    from dargus.workflows.ingest import run_ingest

    file_map: list[tuple] = []
    for fp in sorted(dir_path.iterdir()):
        if not fp.is_file():
            continue
        file_map.append(fp)

    dbase = DBase.global_instance()

    added = 0
    duplicates = 0
    hard_rejects = 0
    errors = 0
    error_details: list[str] = []
    t0 = time.perf_counter()

    for fp in file_map:
        try:
            result = run_ingest({"workflow": "ingest", "source_path": str(fp), "max_rounds": 1})
            added += result.get("n_records", 0)
            duplicates += result.get("n_duplicates", 0)
            errors += result.get("n_errors", 0)
        except Exception as exc:
            errors += 1
            error_details.append(f"{fp.name}: ingest failed — {exc}")

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
