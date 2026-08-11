"""SessionStore — durable archive for ended Sessions (ADR-0005).

Ended Sessions serialize their full Turns→Rounds tree plus metadata to
``{workspace root}/.dargus/sessions/<session_id>.json``. The archive is
**append-only and immutable**: a persisted Session file is never overwritten.
Resume reads back a Session from this archive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dargus.models.session import Session

logger = logging.getLogger(__name__)


def archive_dir(workspace_root: str | Path) -> Path:
    """The per-workspace session archive directory."""
    return Path(workspace_root) / ".dargus" / "sessions"


class SessionStore:
    """Persist and load Session files in a workspace archive."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = str(workspace_root) if workspace_root is not None else ""

    @property
    def workspace_root(self) -> str:
        return self._workspace_root

    # ------------------------------------------------------------------
    # Writing (append-only / immutable)
    # ------------------------------------------------------------------

    def write(self, session: Session) -> Path | None:
        """Serialize *session* to the archive.

        Returns the written path, or ``None`` when the archive is immutable
        (a file for this ``session_id`` already exists) or the workspace root
        is unknown.
        """
        root = session.metadata.workspace_root or self._workspace_root
        if not root:
            logger.warning(
                "SessionStore: no workspace root — skipping persist of %s",
                session.metadata.session_id,
            )
            return None

        target = archive_dir(root) / f"{session.metadata.session_id}.json"
        if target.exists():
            logger.warning(
                "SessionStore: refusing to overwrite archived session %s (append-only)",
                session.metadata.session_id,
            )
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = session.to_dict()
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)  # atomic; the archive entry appears complete
        logger.debug("SessionStore: persisted session %s → %s", session.metadata.session_id, target)
        return target

    # ------------------------------------------------------------------
    # Reading (resume)
    # ------------------------------------------------------------------

    def read(self, session_id: str, workspace_root: str | Path | None = None) -> Session:
        """Load a Session from the archive.

        Raises:
            FileNotFoundError: No archived session with *session_id*.
            ValueError: The archived file is malformed.
        """
        root = workspace_root or self._workspace_root
        target = archive_dir(root) / f"{session_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"No archived session {session_id!r} at {target}")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Archived session {session_id!r} is malformed: {exc}") from exc

    def list_ids(self, workspace_root: str | Path | None = None) -> list[str]:
        """Return the archived session ids under a workspace (sorted)."""
        root = workspace_root or self._workspace_root
        if not root:
            return []
        d = archive_dir(root)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json") if not p.name.endswith(".tmp"))
