"""Shared bootstrap for standalone smoke scripts.

Each ``smoke_*.py`` is run as a script, so its own directory is ``sys.path[0]``
— ``dargus`` is importable only when it is pip-installed in the active env.
Calling :func:`ensure_dargus_on_path` first makes the package importable from a
source checkout under any env (the pytest suite achieves the same via
``pythonpath = ["."]``). Idempotent, no side effects beyond ``sys.path``.
"""

from __future__ import annotations

import os
import sys

#: dargus/tests/smoke/_bootstrap.py -> dargus/tests -> dargus -> repo root
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def ensure_dargus_on_path() -> None:
    """Insert the repo root onto sys.path if it is not already present."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
