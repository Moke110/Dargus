"""Tests for Session persistence — the filesystem seam (ADR-0005, #106).

Ending a Session writes ``{workspace}/.dargus/sessions/<id>.json`` with the
full Turns→Rounds shape and metadata fields; the archive accumulates without
overwrite; resume reads it back correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dargus.models.session import Session, SessionMetadata
from dargus.sessions.store import SessionStore, archive_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tmp workspace root."""
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
    def test_end_writes_full_tree_to_archive(self, workspace: Path):
        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("assess")
        agent._session.add_tool("read_file", params={"path": "/tmp/x"}, output={"ok": True})
        agent._session.add_assistant("done")

        path = agent.end()
        assert path is not None
        assert path == archive_dir(workspace) / f"{agent._session.metadata.session_id}.json"
        assert path.exists()

        data = _load(path)
        assert data["version"] == 1
        turns = data["turns"]
        assert len(turns) == 1
        assert turns[0]["prompt"] == "assess"
        # Full Rounds retained — the tool call is part of the record.
        round_roles = [r["role"] for r in turns[0]["rounds"]]
        assert round_roles == ["assistant", "assistant"]

    def test_metadata_fields_persisted(self, workspace: Path):
        from dargus.agents.base import BaseAgent

        agent = BaseAgent(name="Iris")
        agent._session.metadata.workspace_root = str(workspace)
        agent._session.add_user("q1")
        agent.end()

        ids = SessionStore(str(workspace)).list_ids()
        assert len(ids) == 1
        data = _load(archive_dir(workspace) / f"{ids[0]}.json")
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

    def test_store_without_root_skips(self):
        store = SessionStore("")
        session = _populated_session(root="")
        assert store.write(session) is None

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


class TestSessionDict:
    def test_round_trip_via_to_dict(self):
        session = _populated_session()
        data = session.to_dict()
        rebuilt = Session.from_dict(data)
        assert rebuilt.metadata.session_id == session.metadata.session_id
        assert rebuilt.to_dict() == data
