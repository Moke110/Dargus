from __future__ import annotations

import os
from pathlib import Path


def default_dargus_home() -> Path:
    """Return the Dargus installation home directory."""
    return Path(os.environ.get("DARGUS_HOME", Path.home() / ".dargus"))


def global_dbase_root() -> Path:
    """Return the global D-Base root directory."""
    return default_dargus_home() / "dbase"
