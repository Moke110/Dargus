"""Minimal .env file loader — no external dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ via setdefault.

    Args:
        env_path: Path to .env file. If None, searches: cwd, then workspace
                  root (parent of dargus package).
    """
    if env_path is None:
        dargus_dir = Path(__file__).resolve().parent
        candidates = [
            Path.cwd() / ".env",  # cwd
            dargus_dir.parent / ".env",  # workspace root during dev
        ]
        for candidate in candidates:
            if candidate.exists():
                env_path = candidate
                break
        else:
            return  # no .env found, silently no-op

    env_path = Path(env_path)
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
