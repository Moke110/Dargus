"""WorkspaceGuard — runtime-owned workspace safety foundation (ADR-0001).

The guard enforces an asymmetric workspace boundary:
  - Writes must resolve under a single per-session root (no exceptions).
  - Reads resolve under the root OR any Authorized Path registered by the
    CLI/API layer. Agent/LLM output can never register an Authorized Path.

All decisions canonicalize paths via ``os.path.realpath`` before comparison.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class _WorkspaceError(Exception):
    """Raised internally when a path check fails."""

    def __init__(self, message: str, path: str) -> None:
        super().__init__(message)
        self.path = path


class WorkspaceGuard:
    """Enforce workspace boundaries for file reads and writes.

    Owned by ``DargusRuntime``; constructed once per session.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        authorized_paths: list[str | Path] | None = None,
    ) -> None:
        """Initialise the guard.

        Args:
            root: Workspace root directory. Defaults to the canonicalized
                current working directory.
            authorized_paths: Authorized read paths (canonicalized on entry).
        """
        self._root: str = os.path.realpath(root or os.getcwd())
        self._authorized: set[str] = set()
        for p in authorized_paths or []:
            self._authorized.add(os.path.realpath(p))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> str:
        """Canonical workspace root."""
        return self._root

    @property
    def authorized_paths(self) -> frozenset[str]:
        """Frozen view of currently authorized paths."""
        return frozenset(self._authorized)

    # ------------------------------------------------------------------
    # Authorize (CLI/API only — never reachable from Agent code)
    # ------------------------------------------------------------------

    def authorize(self, path: str | Path) -> None:
        """Register an Authorized Path for reads.

        Only the CLI/API layer may call this from user-supplied arguments.
        Agent/LLM output can never register an Authorized Path — this method
        MUST NOT be exposed to any Agent Tool or Skill.

        Args:
            path: A filesystem path that will be canonicalized and registered.
        """
        canonical = os.path.realpath(path)
        self._authorized.add(canonical)
        logger.debug("Authorized path registered: %s", canonical)

    # ------------------------------------------------------------------
    # Check methods
    # ------------------------------------------------------------------

    def check_write(self, path: str | Path) -> str:
        """Check whether *path* is allowed for writing.

        Returns the canonicalized path on success so callers use the resolved
        form without a second ``realpath`` call.

        Args:
            path: The path the caller intends to write to (can be relative).

        Returns:
            The canonicalized path.

        Raises:
            _WorkspaceError: If the resolved path escapes the workspace root.
        """
        canonical = os.path.realpath(path)
        if not self._is_under(canonical, self._root):
            raise _WorkspaceError(
                f"Write path {path!r} resolves to {canonical!r} — "
                f"outside workspace root {self._root!r}",
                str(path),
            )
        return canonical

    def check_read(self, path: str | Path) -> str:
        """Check whether *path* is allowed for reading.

        Reads are permitted when the resolved path is:
          - under the workspace root, OR
          - an exact match for an Authorized Path, OR
          - under a directory that is an Authorized Path (prefix match).

        Args:
            path: The path the caller intends to read (can be relative).

        Returns:
            The canonicalized path.

        Raises:
            _WorkspaceError: If the resolved path is not in the root and not
                in/under any Authorized Path.
        """
        canonical = os.path.realpath(path)

        if self._is_under(canonical, self._root):
            return canonical

        for auth in self._authorized:
            if canonical == auth or self._is_under(canonical, auth):
                return canonical

        raise _WorkspaceError(
            f"Read path {path!r} resolves to {canonical!r} — "
            f"not under workspace root {self._root!r} "
            f"and not in any Authorized Path",
            str(path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_under(canonical: str, parent: str) -> bool:
        """Return True if *canonical* is under *parent* (or equal to it)."""
        # Canonical paths should never have trailing separators but be safe.
        parent_norm = parent.rstrip(os.sep) + os.sep
        return canonical == parent or canonical.startswith(parent_norm)


def _format_workspace_error(exc: _WorkspaceError) -> dict[str, str]:
    """Format a workspace rejection as a structured error dict.

    This is the documented return shape for file Tools when a guard check
    fails — the Agent sees an error result, not a traceback, so it can
    correct the path and retry.
    """
    return {"error": str(exc), "path": exc.path}
