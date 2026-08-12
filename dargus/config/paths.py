"""Unified configuration file path resolution for Dargus.

The canonical config file is the packaged ``dargus/config/dargus_config.yaml``.
A user override may be placed at ``{Dargus home}/dargus_config.yaml``, where
the Dargus home is the single canonical per-user home (T1).
"""

from __future__ import annotations

from pathlib import Path

from dargus.config.home import user_config_path


def get_config_path() -> Path:
    """Return the canonical Dargus config file path.

    Resolution order:
    1. ``$DARGUS_CONFIG`` environment variable (if set)
    2. ``{Dargus home}/dargus_config.yaml`` (user override)
    3. Packaged ``dargus/config/dargus_config.yaml`` (default)
    """
    import os

    # Environment variable override
    env_path = os.environ.get("DARGUS_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # User home override (resolved through the single Dargus home)
    user_path = user_config_path()
    if user_path.exists():
        return user_path

    # Packaged default
    return Path(__file__).resolve().parent / "dargus_config.yaml"
