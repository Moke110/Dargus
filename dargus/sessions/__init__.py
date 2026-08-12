"""Session archive — durable persistence for ended Sessions (ADR-0005, T2).

The archive is per-user under the Dargus home; reads fall back to the legacy
per-workspace path so no archived Session is lost during migration.
"""

from dargus.sessions.store import (
    SessionStore,
    archive_dir,
    home_archive_dir,
    legacy_archive_dir,
    migrate_legacy_archives,
)

__all__ = [
    "SessionStore",
    "archive_dir",
    "home_archive_dir",
    "legacy_archive_dir",
    "migrate_legacy_archives",
]
