"""D-Base path utilities.

All D-Base paths resolve under the single canonical Dargus home (T1): the
resolver in ``dargus.config.home`` is the one place that decides where the
Dargus home is.
"""

from __future__ import annotations

import os
from pathlib import Path

from dargus.config.home import dargus_home


def default_dargus_home() -> Path:
    """Return the Dargus installation home directory."""
    return dargus_home()


def working_dbase() -> str:
    """Return the active D-Base subdirectory name."""
    return os.environ.get("WORKING_DBASE", "dbase")


def dbase_root() -> Path:
    """Full path: {DARGUS_HOME}/{WORKING_DBASE}."""
    return default_dargus_home() / working_dbase()
