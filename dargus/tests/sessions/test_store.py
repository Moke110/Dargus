"""Tests for Session persistence — the filesystem seam (ADR-0005/T2, #106).

Ending a Session writes ``{Dargus home}/sessions/<id>.json`` — the archive is
per-user, not per-workspace. The archive accumulates without overwrite; reads
try the home archive first and fall back to the legacy
``{workspace}/.dargus/sessions`` path so a pre-migration Session is resumable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dargus.models.session import Session, SessionMetadata
from dargus.sessions.store import (
    SessionStore,
    home_archive_dir,
    legacy_archive_dir,
    migrate_legacy_archives,
)


@pytest.fixture(autouse=True)
def dargus_home(tmp_path: Path, monkeypatch):
    """Point DARGUS_HOME at a tmp dir so tests never touch the real home."""
    home = tmp_path / "dargus_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DARGUS_HOME", str(home))
    return home


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tmp workspace root (legacy archive location)."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _populated_session(session_id: str = "s1", agent: str = "Iris", root: str = "") -> Session:
    session = Session(SessionMetadata(agent=agent, session_id=session_id, workspace_root=root))
    session.add_user("first question")
    session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "data"})
    session.add_assistant("concluded")
    session.close_current_turn()  # a persisted Session has closed Turns
    return session


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# End → write (filesystem seam)
# ------------------------------------------------------------------


class TestPersistOnEnd:
    def test_end_writes_full_tree_to_home_archive(self, workspace: Path):
        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("assess")
        agent._session.add_tool("read_file", params={"path": "/tmp/x"}, output={"ok": True})
        agent._session.add_assistant("done")

        path = agent.end()
        assert path is not None
        assert path == home_archive_dir() / f"{agent._session.metadata.session_id}.json"
        assert path.exists()

        data = _load(path)
        assert data["version"] == 1
        turns = data["turns"]
        assert len(turns) == 1
        assert turns[0]["prompt"] == "assess"
        # Full Rounds retained — the tool call is part of the record.
        round_roles = [r["role"] for r in turns[0]["rounds"]]
        assert round_roles == ["assistant", "assistant"]

    def test_archive_lands_in_home_not_workspace(self, workspace: Path):
        """T2: writes go to {home}/sessions, never the workspace archive."""
        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("q1")
        path = agent.end()
        assert path is not None
        assert legacy_archive_dir(workspace).exists() is False  # nothing written there

    def test_metadata_fields_persisted(self, workspace: Path):
        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("q1")
        agent.end()

        ids = SessionStore(str(workspace)).list_ids()
        assert len(ids) == 1
        data = _load(home_archive_dir() / f"{ids[0]}.json")
        meta = data["metadata"]
        assert meta["agent"] == "Iris"
        assert meta["session_id"] == ids[0]
        assert meta["created"]
        assert meta["closed"] is not None
        assert meta["workspace_root"] == str(workspace)
        assert meta["turn_count"] == 1

    def test_unique_session_ids(self):
        a = Session(SessionMetadata(agent="Iris"))
        b = Session(SessionMetadata(agent="Iris"))
        assert a.metadata.session_id != b.metadata.session_id

    def test_archive_is_append_only(self, workspace: Path):
        """A persisted Session file is never overwritten."""
        store = SessionStore(str(workspace))
        session = _populated_session(session_id="s1", root=str(workspace))
        first = store.write(session)
        assert first is not None
        first_text = first.read_text(encoding="utf-8")

        # Same id, different content — write must refuse.
        session.add_assistant("extra turn")
        assert store.write(session) is None
        assert first.read_text(encoding="utf-8") == first_text

    def test_append_only_refusal_logs_warning(self, workspace: Path, caplog):
        """The genuinely anomalous overwrite still warns (backstop, #109)."""
        import logging

        store = SessionStore(str(workspace))
        session = _populated_session(session_id="warn1", root=str(workspace))
        store.write(session)
        with caplog.at_level(logging.WARNING, logger="dargus.sessions.store"):
            assert store.write(session) is None
        assert "refusing to overwrite archived session" in caplog.text

    def test_successful_persist_logs_debug_not_info(self, workspace: Path, caplog):
        """Persist is bookkeeping, not user-facing: INFO → DEBUG (#109)."""
        import logging

        store = SessionStore(str(workspace))
        with caplog.at_level(logging.DEBUG, logger="dargus.sessions.store"):
            store.write(_populated_session(session_id="dbg1", root=str(workspace)))
        assert "persisted session" in caplog.text
        # No INFO-level "persisted session …" line at exit.
        assert not any(
            r.levelno >= logging.INFO and "persisted session" in r.message for r in caplog.records
        )

    def test_write_without_workspace_root_persists_to_home(self):
        """T2: a Session persists to the home archive even with no root."""
        store = SessionStore("")
        session = _populated_session(root="")
        path = store.write(session)
        assert path is not None
        assert path == home_archive_dir() / f"{session.metadata.session_id}.json"
        assert path.exists()

    def test_atexit_safety_net_persists(self, workspace: Path):
        """The atexit-registered close persists a Session whose workspace is
        known from the runtime."""
        from dargus.agents.base import BaseAgent
        from dargus.runtime.context import DargusRuntime

        runtime = DargusRuntime(config={"workspace_root": str(workspace)})
        assert runtime.workspace_guard.root == str(workspace)

        agent = BaseAgent(name="Iris")
        agent._runtime = runtime
        agent._session.add_user("q1")
        agent._session.add_assistant("reply")

        # Resolving the session seeds the workspace root from the guard.
        agent._resolve_session({})
        assert agent._session.metadata.workspace_root == str(workspace)

        path = agent.end()
        assert path is not None and path.exists()

    def test_double_persist_is_silent_noop(self, workspace: Path, caplog):
        """The atexit re-persist returns None silently (no append-only warning).

        (#109: the REPL persists on quit and the atexit safety net re-persists;
        the second write is a true no-op, not a flagged archive collision.)
        """
        import logging

        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("q1")
        agent._session.add_assistant("reply")

        first = agent.end()
        assert first is not None

        with caplog.at_level(logging.WARNING, logger="dargus.agents.base"):
            second = agent.end()
        assert second is None  # silent no-op
        assert "refusing to overwrite" not in caplog.text


# ------------------------------------------------------------------
# Store round-trip (model seam)
# ------------------------------------------------------------------


class TestStoreRoundTrip:
    def test_write_then_read_round_trips_tree(self, workspace: Path):
        store = SessionStore(str(workspace))
        session = _populated_session(session_id="rt1", root=str(workspace))
        store.write(session)

        loaded = store.read("rt1")
        assert loaded.metadata.session_id == "rt1"
        assert loaded.metadata.agent == "Iris"
        assert loaded.metadata.turn_count == 1
        assert len(loaded.turns) == 1
        assert loaded.turns[0].prompt == "first question"
        assert loaded.turns[0].rounds[0].tool_name == "read_file"
        assert loaded.turns[0].rounds[0].tool_output == {"content": "data"}
        assert loaded.turns[0].rounds[1].text == "concluded"
        # A loaded Session is a full record, not a projection.
        assert len(loaded.messages) == 3

    def test_read_missing_raises(self, workspace: Path):
        store = SessionStore(str(workspace))
        with pytest.raises(FileNotFoundError):
            store.read("nope")

    def test_list_ids(self, workspace: Path):
        store = SessionStore(str(workspace))
        store.write(_populated_session(session_id="a", root=str(workspace)))
        store.write(_populated_session(session_id="b", root=str(workspace)))
        assert store.list_ids() == ["a", "b"]

    def test_load_path_independent_projection(self, workspace: Path):
        """A resumed Session projects coarse for loaded (closed) turns —
        the structural rule, independent of how the Session was loaded."""
        store = SessionStore(str(workspace))
        session = _populated_session(session_id="proj", root=str(workspace))
        store.write(session)

        loaded = store.read("proj")
        msgs = loaded.projection()
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "first question"
        assert msgs[1].content == "concluded"
        assert "read_file" not in " ".join(m.content for m in msgs)


# ------------------------------------------------------------------
# Dual-read fallback (T2)
# ------------------------------------------------------------------


class TestLegacyFallback:
    def test_read_falls_back_to_legacy_workspace_archive(self, workspace: Path):
        """A session archived before the move is still resumable."""
        legacy = legacy_archive_dir(workspace)
        legacy.mkdir(parents=True, exist_ok=True)
        session = _populated_session(session_id="legacy-1", root=str(workspace))
        (legacy / "legacy-1.json").write_text(
            json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

        store = SessionStore(str(workspace))
        loaded = store.read("legacy-1")
        assert loaded.metadata.session_id == "legacy-1"
        assert loaded.metadata.agent == "Iris"
        assert home_archive_dir().exists() is False or "legacy-1.json" not in [
            p.name for p in home_archive_dir().glob("*.json")
        ]

    def test_home_archive_wins_over_legacy(self, workspace: Path):
        """When both archives hold an id, the home copy is authoritative."""
        store = SessionStore(str(workspace))
        store.write(_populated_session(session_id="dup", root=str(workspace), agent="Home"))

        legacy = legacy_archive_dir(workspace)
        legacy.mkdir(parents=True, exist_ok=True)
        session = _populated_session(session_id="dup", root=str(workspace), agent="Legacy")
        (legacy / "dup.json").write_text(
            json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

        loaded = store.read("dup")
        assert loaded.metadata.agent == "Home"

    def test_list_ids_merges_home_and_legacy(self, workspace: Path):
        store = SessionStore(str(workspace))
        store.write(_populated_session(session_id="home-a", root=str(workspace)))
        store.write(_populated_session(session_id="home-b", root=str(workspace)))

        legacy = legacy_archive_dir(workspace)
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "legacy-c.json").write_text("{}", encoding="utf-8")

        assert store.list_ids() == ["home-a", "home-b", "legacy-c"]

    def test_migrate_legacy_archives_copies_into_home(self, workspace: Path):
        legacy = legacy_archive_dir(workspace)
        legacy.mkdir(parents=True, exist_ok=True)
        for sid in ("m1", "m2"):
            session = _populated_session(session_id=sid, root=str(workspace))
            (legacy / f"{sid}.json").write_text(
                json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8"
            )

        migrated = migrate_legacy_archives([workspace])
        assert migrated == 2
        assert (home_archive_dir() / "m1.json").exists()
        assert (home_archive_dir() / "m2.json").exists()

        # The legacy originals are left in place (non-destructive).
        assert (legacy / "m1.json").exists()

    def test_migrate_dedupes_by_session_id_never_overwrites(self, workspace: Path):
        """Migration skips ids already in the home archive."""
        store = SessionStore(str(workspace))
        existing = _populated_session(session_id="shared", root=str(workspace), agent="Home")
        store.write(existing)

        legacy = legacy_archive_dir(workspace)
        legacy.mkdir(parents=True, exist_ok=True)
        older = _populated_session(session_id="shared", root=str(workspace), agent="Legacy")
        (legacy / "shared.json").write_text(
            json.dumps(older.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

        migrated = migrate_legacy_archives([workspace])
        assert migrated == 0  # nothing new to copy
        loaded = store.read("shared")
        assert loaded.metadata.agent == "Home"  # never overwritten

    def test_migrate_no_legacy_archive_is_noop(self, workspace: Path):
        assert migrate_legacy_archives([workspace]) == 0


class TestSessionDict:
    def test_round_trip_via_to_dict(self):
        session = _populated_session()
        data = session.to_dict()
        rebuilt = Session.from_dict(data)
        assert rebuilt.metadata.session_id == session.metadata.session_id
        assert rebuilt.to_dict() == data
