"""SessionStore — durable archive for ended Sessions (ADR-0005, revised T2).

Ended Sessions serialize their full Turns→Rounds tree plus metadata to
``{Dargus home}/sessions/<session_id>.json`` — the archive is **per-user**,
not per-workspace (T2). The archive is **append-only and immutable**: a
persisted Session file is never overwritten. Reads try the home archive first
and fall back to the legacy ``{workspace root}/.dargus/sessions`` path so no
archived Session is lost during the migration to the per-user home.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from dargus.config.home import sessions_dir
from dargus.models.session import Session

logger = logging.getLogger(__name__)


def home_archive_dir() -> Path:
    """The per-user session archive directory (``{home}/sessions``)."""
    return sessions_dir()


def legacy_archive_dir(workspace_root: str | Path) -> Path:
    """The legacy per-workspace session archive directory."""
    return Path(workspace_root) / ".dargus" / "sessions"


def archive_dir() -> Path:
    """The session archive directory — the per-user home archive (T2)."""
    return home_archive_dir()


class SessionStore:
    """Persist and load Session files in the per-user home archive.

    ``workspace_root`` is retained for **read-only legacy fallback**: a
    Session archived under the old ``{workspace}/.dargus/sessions`` layout is
    still found (and resumable) until migration moves it into the home.
    """

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = str(workspace_root) if workspace_root is not None else ""

    @property
    def workspace_root(self) -> str:
        return self._workspace_root

    # ------------------------------------------------------------------
    # Writing (append-only / immutable)
    # ------------------------------------------------------------------

    def write(self, session: Session) -> Path | None:
        """Serialize *session* to the home archive.

        The archive location is the per-user home — ``workspace_root``
        metadata no longer determines where a Session lands (T2).

        Returns the written path, or ``None`` when the archive is immutable
        (a file for this ``session_id`` already exists).
        """
        target = home_archive_dir() / f"{session.metadata.session_id}.json"
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

        Tries the home archive first, then the legacy ``workspace_root``
        archive, so a pre-migration Session is still resumable.

        Raises:
            FileNotFoundError: No archived session with *session_id*.
            ValueError: The archived file is malformed.
        """
        target = self._find(session_id, workspace_root)
        if target is None:
            raise FileNotFoundError(
                f"No archived session {session_id!r} in the Dargus home archive"
            )
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Archived session {session_id!r} is malformed: {exc}") from exc

    def _find(self, session_id: str, workspace_root: str | Path | None = None) -> Path | None:
        """The archive path for *session_id*, or ``None`` if it does not exist."""
        candidates = [home_archive_dir()]
        root = workspace_root or self._workspace_root
        if root:
            candidates.append(legacy_archive_dir(root))
        for directory in candidates:
            target = directory / f"{session_id}.json"
            if target.exists():
                return target
        return None

    def list_ids(self, workspace_root: str | Path | None = None) -> list[str]:
        """Return the archived session ids (home + legacy, sorted)."""
        root = workspace_root or self._workspace_root
        ids: set[str] = set()

        def _collect(directory: Path) -> None:
            if directory.exists():
                ids.update(p.stem for p in directory.glob("*.json") if not p.name.endswith(".tmp"))

        _collect(home_archive_dir())
        if root:
            _collect(legacy_archive_dir(root))
        return sorted(ids)


def migrate_legacy_archives(
    workspace_roots: list[str | Path],
    target_home: str | Path | None = None,
) -> int:
    """Copy-merge legacy per-workspace session archives into the home archive.

    For every ``{workspace}/.dargus/sessions/<id>.json``, the file is copied
    into the home archive **only if** no file for that ``session_id`` exists
    there yet — deduped by ``session_id``, never overwriting. The legacy
    originals are left in place.

    Returns:
        The number of session files migrated into the home archive.
    """
    target = Path(target_home) / "sessions" if target_home else home_archive_dir()
    target.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for root in workspace_roots:
        source = legacy_archive_dir(root)
        if not source.exists():
            continue
        for path in sorted(source.glob("*.json")):
            if path.name.endswith(".tmp"):
                continue
            dest = target / path.name
            if dest.exists():
                logger.debug("migrate: %s already in home archive — skipping", path.name)
                continue
            shutil.copy2(path, dest)
            migrated += 1
            logger.debug("migrate: %s → %s", path, dest)
    return migrated
