"""Minimal .env file loader — no external dependency."""

from __future__ import annotations

import os
from pathlib import Path


def use_local_model_cost_map() -> None:
    """Force LiteLLM to use its bundled local model cost map (ADR-0007).

    On its first import LiteLLM otherwise tries to fetch the model cost map
    from GitHub (5s timeout), and logs a WARNING when the fetch fails — which
    the REPL rendered as an error line on the first message. The local backup
    is what LiteLLM falls back to anyway, so Dargus skips the remote fetch.

    Must be called before ``import litellm``. ``setdefault`` means a user who
    exports ``LITELLM_LOCAL_MODEL_COST_MAP`` themselves still wins.
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")


def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env file(s) into os.environ via setdefault.

    Args:
        env_path: Path to a specific .env file. If None, loads from the
                  Dargus home ``{home}/.env`` first, then searches cwd and
                  the workspace root (dev) — the home is authoritative
                  (T3), while cwd/workspace stay as backward-compatible
                  fallbacks so existing keys keep resolving.
    """
    if env_path is None:
        from dargus.config.home import env_path as home_env_path

        dargus_dir = Path(__file__).resolve().parent
        candidates = [
            home_env_path(),  # {Dargus home}/.env — home-first (T3)
            Path.cwd() / ".env",  # cwd
            dargus_dir.parent / ".env",  # workspace root during dev
        ]
        for candidate in candidates:
            if candidate.exists():
                _load_env_file(candidate)
        return  # all candidates loaded home-first; no-op if none exist

    _load_env_file(Path(env_path))


def _load_env_file(env_path: Path) -> None:
    """Load a single .env file into os.environ via setdefault."""
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            os.environ.setdefault(key, value)


def write_dotenv(key: str, value: str, env_path: str | Path | None = None) -> str:
    """Write or update a KEY=VALUE line in a .env file.

    Args:
        key: the environment variable name
        value: the value to assign
        env_path: path to .env file. If None, writes to ``{home}/.env``
            (the canonical secrets location, T3).

    Returns:
        The path to the .env file that was written.
    """
    if env_path is None:
        from dargus.config.home import env_path as home_env_path

        env_path = home_env_path()
    else:
        env_path = Path(env_path)

    lines: list[str] = []
    found = False
    prefix = f"{key}="

    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"# {key}="):
                    lines.append(f"{prefix}{value}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"{prefix}{value}\n")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    return str(env_path)
