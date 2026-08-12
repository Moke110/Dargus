"""Dargus home — the single canonical per-user home (T1).

All per-user Dargus paths resolve through one resolver: configuration,
secrets, the D-Base, and the session archive. ``$DARGUS_HOME`` overrides the
default ``~/.dargus/``; every path module (config, dbase, sessions, dotenv)
consumes this module so their paths agree.
"""

from __future__ import annotations

import os
from pathlib import Path


def dargus_home() -> Path:
    """Return the canonical Dargus home directory.

    Resolution order:
    1. ``$DARGUS_HOME`` (explicit override, wins)
    2. a chosen home recorded by ``dargus setup`` (``~/.dargus-home``)
    3. ``~/.dargus`` (the default)
    """
    env = os.environ.get("DARGUS_HOME")
    if env:
        return Path(env).expanduser().resolve()

    chosen = _chosen_home()
    if chosen is not None:
        return chosen

    return Path.home() / ".dargus"


def _chosen_home() -> Path | None:
    """The home recorded by ``dargus setup``, if any."""
    pointer = Path.home() / ".dargus-home"
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def record_home(home: str | Path) -> Path:
    """Record the Dargus home chosen at setup so it survives across runs.

    Only meaningful when the user picked a non-default location; recording
    the default is harmless (resolution falls back to the default anyway).
    """
    pointer = Path.home() / ".dargus-home"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(home), encoding="utf-8")
    return Path(home).expanduser().resolve()


def user_config_path() -> Path:
    """The user config file under the Dargus home."""
    return dargus_home() / "dargus_config.yaml"


def env_path() -> Path:
    """The secrets file (``{home}/.env``) under the Dargus home."""
    return dargus_home() / ".env"


def sessions_dir() -> Path:
    """The per-user session archive directory (``{home}/sessions``)."""
    return dargus_home() / "sessions"


def is_initialized() -> bool:
    """True when Dargus home holds a user config (setup has run).

    Setup's marker is the user config file: its presence means the machine
    has been initialised, which is what the first-run guards gate on.
    """
    return user_config_path().exists()
