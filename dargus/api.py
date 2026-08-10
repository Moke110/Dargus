"""Dargus public API facade.

All adapters (CLI, Claude Code skill) MUST import only from this module.
No adapter imports anything deeper than ``dargus.api``.
"""

from __future__ import annotations

import logging
from typing import Any

from dargus.iris.commander import Iris

logger = logging.getLogger(__name__)


#: Process-lifetime runtime accessor. The first API call bootstraps; every
#: subsequent call reuses the same DargusRuntime (holding at most one live
#: Iris) so the live Session survives across ask/status turns.
_RUNTIME_CACHE: Any | None = None


def _get_runtime() -> Any:
    """Return the process-lifetime DargusRuntime, bootstrapping it once.

    Bootstraps lazily on first call and caches the runtime. The runtime
    starts healthy and entry points refuse new sessions while unhealthy —
    there is no silent fallback to a runtime-less path.
    """
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        _RUNTIME_CACHE.ensure_healthy()
        return _RUNTIME_CACHE

    from dargus.runtime.bootstrap import bootstrap

    try:
        runtime = bootstrap()
    except Exception as exc:
        logger.error("API: runtime bootstrap failed: %s", exc)
        raise RuntimeError(f"DargusRuntime failed to start: {exc} — refusing new session") from exc
    runtime.ensure_healthy()
    _RUNTIME_CACHE = runtime
    _register_atexit_persist()
    return runtime


def _register_atexit_persist() -> None:
    """Register the atexit safety net: persist the live Iris's Session on
    process exit (never ``__del__``). Guarded so repeated registration is a
    no-op."""
    import atexit

    if getattr(_register_atexit_persist, "_registered", False):
        return
    _register_atexit_persist._registered = True  # type: ignore[attr-defined]
    atexit.register(_persist_live_session_on_exit)


def _persist_live_session_on_exit() -> None:
    """Persist the live Session when the process exits (atexit safety net).

    Idempotent: the archive is append-only, so re-persisting an already
    written Session is a no-op.
    """
    global _RUNTIME_CACHE
    runtime = _RUNTIME_CACHE
    if runtime is None:
        return
    factory = getattr(runtime, "agent_factory", None)
    if factory is None:
        return
    iris = getattr(factory, "_iris_cache", None)
    if iris is None:
        return
    end = getattr(iris, "end", None)
    if callable(end):
        try:
            end()
        except Exception:
            logger.exception("API: failed to persist live session at exit")


def _create_iris_with_lm() -> Iris:
    """Create Iris through the runtime's AgentFactory.

    Uses the process-lifetime runtime so the same DargusRuntime and the same
    cached Iris are reused across API calls.
    """
    return _get_runtime().agent_factory.iris()


def status() -> dict[str, Any]:
    """Report global D-Base status.

    Returns:
        Dict with ``dargus_home``, ``n_records``, ``n_templates``.
    """
    iris = _create_iris_with_lm()
    return iris.status()


# ---------------------------------------------------------------------------
# Session / Environment
# ---------------------------------------------------------------------------


def init() -> None:
    """Initialize Dargus session — load .env and configure logging."""
    import logging

    from dargus._env import load_dotenv

    load_dotenv()

    # Route LiteLLM logs with a ``[LiteLLM]`` prefix so they are
    # visually distinct from Iris output.  The REPL overrides this
    # with a Rich handler for coloured output.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    if not litellm_logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.WARNING)
        h.setFormatter(logging.Formatter("[LiteLLM] %(message)s"))
        litellm_logger.addHandler(h)


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
    if iris._reasoning_llm is None:
        from dargus.runtime.errors import NoLLMConfiguredError

        raise NoLLMConfiguredError()

    # Run Iris through the unified PRA loop. The live Iris owns its Session
    # (ADR-0005): each ask() appends a fresh turn, so follow-ups like "yes"
    # resolve against prior dialogue (SPEC-B).
    report = iris.run({"query": query})

    # Extract text response from the AgentReport
    if report.findings and isinstance(report.findings[-1], str):
        return report.findings[-1]

    # Fallback: extract from the last reason-phase trace
    for trace in reversed(report.call_trace):
        if trace.phase == "reason" and trace.output_summary:
            return trace.output_summary

    return "Iris processed your request."


# ---------------------------------------------------------------------------
# Session lifecycle (ADR-0005)
# ---------------------------------------------------------------------------


def new_session() -> str:
    """Start a fresh empty Session (the ``/new`` entry point).

    The current live Session is persisted-then-ended first (the swap verb),
    so switching contexts never silently discards work. Returns the new
    Session's id.
    """
    runtime = _get_runtime()
    iris = runtime.agent_factory.swap(hydrate=None)
    return iris._session.metadata.session_id


def resume_session(session_id: str) -> str:
    """Resume an archived Session under a fresh identity (the ``/resume``
    entry point).

    Persists-then-ends the current live Session, then hydrates the chosen
    archived Session into a fresh Iris with a **fresh** ``session_id``. All
    loaded Turns project coarse (the structural rule); the archived original
    is never mutated.

    Args:
        session_id: The archived Session to resume.

    Returns:
        The new (resumed) Session's fresh id.

    Raises:
        FileNotFoundError: No archived Session with *session_id* in the
            current workspace.
    """
    runtime = _get_runtime()
    root = _workspace_root(runtime)
    from dargus.sessions.store import SessionStore

    loaded = SessionStore(root).read(session_id)

    # Fresh identity — resume continues prior work as a *new* Session.
    from dargus.models.session import new_session_id

    loaded.metadata.session_id = new_session_id()
    loaded.metadata.closed = None

    iris = runtime.agent_factory.swap(hydrate=loaded)
    return iris._session.metadata.session_id


def end_session() -> str | None:
    """Explicitly end and persist the live Session (runtime exit).

    Idempotent — the archive is append-only. Returns the persisted Session
    id, or ``None`` when there is no live Session.
    """
    runtime = _RUNTIME_CACHE
    if runtime is None:
        return None
    factory = getattr(runtime, "agent_factory", None)
    if factory is None:
        return None
    iris = getattr(factory, "_iris_cache", None)
    if iris is None:
        return None
    session_id = iris._session.metadata.session_id
    iris.close()
    return session_id


def _workspace_root(runtime: Any) -> str:
    """Resolve the workspace root off the runtime (ADR-0005)."""
    return runtime.workspace_root


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

    # Read from models.reasoning block (single source of truth)
    reasoning_cfg = (cfg.get("models") or {}).get("reasoning", {})
    return {
        "provider": reasoning_cfg.get("provider", "openai"),
        "model": reasoning_cfg.get("model", ""),
        "base_url": reasoning_cfg.get("base_url", "https://api.deepseek.com"),
        "temperature": reasoning_cfg.get("temperature", 0.0),
        "max_tokens": reasoning_cfg.get("max_tokens", 2048),
        "has_api_key": has_api_key(),
    }


def save_llm_config(model: str, base_url: str, provider: str = "openai") -> None:
    """Save LLM configuration to config file.

    Args:
        model: Model name.
        base_url: API base URL.
        provider: Provider name (default: openai).
    """
    from dargus.config.paths import get_config_path

    config_path = get_config_path()

    # Load existing config
    cfg: dict = {}
    if config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    # Write into models.reasoning block (single source of truth)
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


def test_llm_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Test LLM connection using the same LiteLLM call path that Iris uses.

    Args:
        provider: LiteLLM provider name (e.g. ``deepseek``, ``openai``).
        model: Model name.
        base_url: API base URL.
        api_key: Optional API key (uses env var if not provided).

    Returns:
        Dict with keys: ok, model, latency_ms, error.
    """
    import os
    import time

    import litellm

    key = api_key or os.environ.get("DARGUS_LLM_API_KEY")

    t0 = time.monotonic()
    try:
        model_id = f"{provider}/{model}"
        kwargs: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with just: OK"}],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["api_base"] = base_url

        response = litellm.completion(**kwargs)
        response.choices[0].message.content  # verify response is valid
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
