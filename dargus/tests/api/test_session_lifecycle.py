"""Tests for the API session lifecycle — the public API seam (ADR-0005, #107).

Drives ``api.ask()`` / ``api.new_session()`` / ``api.resume_session(id)``.
Asserts: follow-ups resolve against prior turns; ``/new`` yields a fresh
session; ``/resume`` yields prior turns (coarse) under a new identity; only
one live session exists across swaps.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dargus.models.reasoning import LLMResponse, LLMUsage, Message, ReasoningLLM
from dargus.runtime.context import DargusRuntime


class _ScriptedBackend:
    """ReasoningBackend returning queued PRA responses, then a default."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def chat(self, messages: list[Message], options=None) -> LLMResponse:
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = '{"action": "text", "text": "done"}'
        return LLMResponse(content=content, usage=LLMUsage(), model="fake")


@pytest.fixture
def wire_runtime(tmp_path: Path, monkeypatch):
    """Wire ``dargus.api._RUNTIME_CACHE`` to a fresh test runtime.

    Each test calls ``wire_runtime(responses=[...])`` to control the scripted
    LLM, and the wiring is torn down afterwards.
    """
    import dargus.api as api

    runtimes: list[DargusRuntime] = []

    def _wire(responses: list[str]) -> DargusRuntime:
        import json

        rt = DargusRuntime(config={"workspace_root": str(tmp_path)})
        scripted = [
            json.dumps({"action": "text", "text": r}) if not r.startswith("{") else r
            for r in responses
        ]
        rt.reasoning_llm = ReasoningLLM(backend=_ScriptedBackend(scripted))
        monkeypatch.setattr(api, "_RUNTIME_CACHE", rt)
        runtimes.append(rt)
        return rt

    yield _wire
    # The atexit safety net must not fire with a stale runtime.
    monkeypatch.setattr(api, "_RUNTIME_CACHE", None)


def _archive_ids(rt: DargusRuntime) -> list[str]:
    from dargus.sessions.store import SessionStore

    return SessionStore(str(rt.workspace_guard.root)).list_ids()


def _live(rt: DargusRuntime):
    return rt.agent_factory._iris_cache


class TestNewSession:
    def test_new_session_persists_then_starts_fresh(self, tmp_path, wire_runtime):
        import dargus.api as api

        rt = wire_runtime(["first reply", "fresh-session reply"])

        # One turn in the live session.
        assert "first reply" in api.ask("hello")

        old_id = _live(rt)._session.metadata.session_id
        new_id = api.new_session()

        # The old session was persisted before being ended.
        assert old_id in _archive_ids(rt)
        # A fresh empty session exists now.
        assert new_id != old_id
        fresh = _live(rt)._session
        assert fresh.metadata.session_id == new_id
        assert fresh.metadata.turn_count == 0
        assert len(fresh.turns) == 0

        # The fresh session answers independently.
        assert "fresh-session reply" in api.ask("new conversation")

    def test_one_live_iris_across_swap(self, wire_runtime):
        import dargus.api as api

        rt = wire_runtime(["first reply"])
        api.ask("hello")
        before = _live(rt)
        api.new_session()
        after = _live(rt)
        assert before is not after  # swapped to a fresh Iris
        assert _live(rt) is after  # one live Iris


class TestResumeSession:
    def test_resume_continues_prior_turns_under_new_identity(self, wire_runtime):
        import dargus.api as api

        rt = wire_runtime(["first reply", "resumed reply"])
        api.ask("what is aspirin?")
        archived_id = _live(rt)._session.metadata.session_id

        # Swap away — the archived session is now on disk.
        api.new_session()

        # Resume the archived session.
        resumed_id = api.resume_session(archived_id)
        assert resumed_id != archived_id  # fresh identity
        resumed = _live(rt)._session
        assert resumed.metadata.session_id == resumed_id
        # Loaded Turns are there (all closed → coarse projection).
        assert len(resumed.turns) == 1

        # The archived original is untouched.
        assert archived_id in _archive_ids(rt)

        # Continuing the resumed session works; the prior turn is recalled
        # coarsely, so a follow-up resolves.
        assert "resumed reply" in api.ask("yes, go ahead")
        assert len(resumed.turns) == 2

    def test_resume_unknown_id_raises(self, wire_runtime):
        import dargus.api as api

        wire_runtime([])
        with pytest.raises(FileNotFoundError):
            api.resume_session("no-such-session")

    def test_resume_projects_prior_turns_coarse(self, wire_runtime):
        """A resumed session's loaded turns project coarse (prompt + final
        reply only) — the structural rule."""
        import dargus.api as api

        rt = wire_runtime(["first reply"])
        api.ask("first question")
        archived_id = _live(rt)._session.metadata.session_id
        api.new_session()
        api.resume_session(archived_id)

        # The in-memory projection of the loaded turns is coarse.
        resumed = _live(rt)._session
        msgs = resumed.projection()
        # First turn: user prompt (JSON) + assistant reply, no tool noise.
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[1].content == "first reply"


class TestEndSession:
    def test_end_session_persists_live(self, wire_runtime):
        import dargus.api as api

        rt = wire_runtime(["first reply"])
        api.ask("hello")
        session_id = _live(rt)._session.metadata.session_id
        assert api.end_session() == session_id
        assert session_id in _archive_ids(rt)

    def test_end_session_idempotent(self, wire_runtime):
        import dargus.api as api

        wire_runtime(["first reply"])
        api.ask("hello")
        api.end_session()
        # Second end is a no-op (append-only archive) but still returns the id.
        assert api.end_session() is not None


class TestInterruptedTool:
    def test_interrupted_tool_settles_as_error_entry(self, tmp_path, wire_runtime):
        """#105: an interrupted/failed tool call settles as an error Round in
        the Session — at the API seam."""
        import dargus.api as api

        # A scripted LLM that requests read_file but the tool execution will
        # fail (tool not in PERMITTED_TOOLS → "not permitted" error).
        rt = wire_runtime(['{"action": "tool_call", "tool": "write_file", "params": {}}'])
        api.ask("do the thing")

        session = _live(rt)._session
        tool_rounds = [m for m in session.messages if m.tool_name is not None]
        assert len(tool_rounds) == 1
        # The failed call settles as an error in the record, not dropped.
        assert "not permitted" in str(tool_rounds[0].tool_output)
        assert tool_rounds[0].tool_name == "write_file"

        # The turn is closed (no text reply) and the error is retained in
        # the archive record.
        assert session.turns[-1].closed is True
        api.end_session()
        archived = _archive_ids(rt)
        assert len(archived) == 1


class TestAtexitSafetyNet:
    def test_register_atexit_is_guarded_noop(self, monkeypatch):
        """Registering the atexit handler twice only registers once."""
        import dargus.api as api

        setattr(api._register_atexit_persist, "_registered", False)
        with patch("atexit.register") as mock_register:
            api._register_atexit_persist()
            api._register_atexit_persist()
            mock_register.assert_called_once()

    def test_exit_handler_persists_live_session(self, tmp_path, wire_runtime):
        """The atexit handler persists the live Iris Session (never
        ``__del__``)."""
        import dargus.api as api

        rt = wire_runtime(["first reply"])
        api.ask("hello")
        session_id = _live(rt)._session.metadata.session_id

        api._persist_live_session_on_exit()
        assert session_id in _archive_ids(rt)

    def test_exit_handler_noop_without_runtime(self):
        import dargus.api as api

        api._persist_live_session_on_exit()  # must not raise
