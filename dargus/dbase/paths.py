"""D-Base path utilities."""

from __future__ import annotations

import os
from pathlib import Path


def default_dargus_home() -> Path:
    """Return the Dargus installation home directory."""
    return Path(os.environ.get("DARGUS_HOME", Path.home() / ".dargus"))


def working_dbase() -> str:
    """Return the active D-Base subdirectory name."""
    return os.environ.get("WORKING_DBASE", "dbase")


def dbase_root() -> Path:
    """Full path: {DARGUS_HOME}/{WORKING_DBASE}."""
    return default_dargus_home() / working_dbase()
