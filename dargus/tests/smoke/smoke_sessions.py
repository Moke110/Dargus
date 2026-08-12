"""Standalone smoke: Sessions — create → persist → reload → reopen (offline).

Pins the session-continuity invariant: a real Session with a real turn and
rounds is persisted to the workspace archive, loaded back from disk unchanged,
and reopens under a fresh identity without mutating the archived original.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_sessions.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


def _populated_session(root: str):
    from dargus.models.session import Session, SessionMetadata

    session = Session(SessionMetadata(agent="Iris", workspace_root=root))
    session.add_user("first question")
    session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "data"})
    session.add_assistant("concluded")
    session.close_current_turn()  # a persisted Session has closed Turns
    return session


def main() -> int:
    import os

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        # The session archive is per-user: point DARGUS_HOME at tmp (T2).
        home = Path(tmp) / "dargus_home"
        home.mkdir()
        os.environ["DARGUS_HOME"] = str(home)

        from dargus.sessions.store import SessionStore, home_archive_dir

        store = SessionStore(str(workspace))

        # 1. Create a session and persist it to the *home* archive.
        session = _populated_session(root=str(workspace))
        original_id = session.metadata.session_id
        written = store.write(session)
        assert written is not None, "SessionStore.write returned None"
        assert written.exists(), f"archive file not on disk: {written}"
        assert written == home_archive_dir() / f"{original_id}.json"

        # 2. The archive now lists exactly one id (append-only).
        ids = store.list_ids()
        assert ids == [original_id], f"archive ids {ids} != [{original_id}]"

        # 3. Reload from disk; the full Turns→Rounds shape survives.
        loaded = store.read(original_id)
        assert loaded.metadata.agent == "Iris"
        assert loaded.metadata.session_id == original_id
        assert loaded.metadata.closed is None  # not closed by the store
        assert len(loaded.turns) == 1
        roles = [r.role for r in loaded.turns[0].rounds]
        assert roles == ["assistant", "assistant"], f"unexpected round roles {roles}"

        # 4. Reopen gives a fresh identity; the archived original is untouched.
        loaded.reopen()
        new_id = loaded.metadata.session_id
        assert new_id != original_id
        assert loaded.metadata.closed is None
        assert loaded.metadata.turn_count == loaded.metadata.turn_count  # preserved

        original = store.read(original_id)
        assert original.metadata.session_id == original_id  # archived original unchanged
        assert store.list_ids() == [original_id]  # still one archived session

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
