"""Tests for the Session aggregate, Turn, and Round model (ADR-0005).

Exercises the external behavior of the model: appending preserves order,
``projection()`` projects each role correctly, and an assistant Round
records at most one tool payload.
"""

from __future__ import annotations

from dargus.models.session import (
    Round,
    Session,
    SessionMetadata,
    Turn,
)


class TestRound:
    def test_user_constructor_sets_role_and_text(self):
        r = Round.user("hello")
        assert r.role == "user"
        assert r.text == "hello"

    def test_assistant_constructor_defaults_no_tool(self):
        r = Round.assistant("done")
        assert r.role == "assistant"
        assert r.tool_name is None
        assert r.tool_error is None

    def test_synthetic_constructor(self):
        r = Round.synthetic("subagent result")
        assert r.role == "synthetic"
        assert r.text == "subagent result"

    def test_assistant_at_most_one_tool_payload(self):
        """An assistant Round carries one tool call + one result payload."""
        r = Round(
            role="assistant",
            tool_name="read_file",
            tool_params={"path": "/tmp/x"},
            tool_output={"ok": True},
        )
        assert r.tool_name is not None
        assert r.tool_output is not None
        assert r.role == "assistant"

    def test_projects_user_role(self):
        r = Round.user("hi")
        llm = r.as_llm_message()
        assert llm.role == "user"
        assert llm.content == "hi"

    def test_projects_synthetic_role(self):
        r = Round.synthetic("delegation")
        llm = r.as_llm_message()
        assert llm.role == "synthetic"
        assert llm.content == "delegation"

    def test_projects_assistant_with_tool_renders_round(self):
        r = Round(
            role="assistant",
            tool_name="read_file",
            tool_params={"path": "/tmp/x"},
            tool_output={"content": "data"},
        )
        llm = r.as_llm_message()
        assert llm.role == "assistant"
        assert "[tool_call] read_file" in llm.content
        assert "path" in llm.content
        assert "data" in llm.content

    def test_projects_assistant_with_tool_error(self):
        r = Round(role="assistant", tool_name="read_file", tool_params={}, tool_error="boom")
        llm = r.as_llm_message()
        assert "[tool_error] boom" in llm.content


class TestTurn:
    def test_summary_is_prompt_plus_final_reply(self):
        turn = Turn(prompt="q1")
        turn.rounds.append(Round.tool("read_file", {"path": "/x"}, output={"ok": True}))
        turn.rounds.append(Round.assistant("concluded"))
        assert turn.summary == ("q1", "concluded")

    def test_final_reply_is_last_non_tool_assistant(self):
        turn = Turn(prompt="q1")
        turn.rounds.append(Round.assistant("interim"))
        turn.rounds.append(Round.tool("read_file", {"path": "/x"}, output={"ok": True}))
        assert turn.final_reply == "interim"

    def test_final_reply_empty_when_no_text_round(self):
        turn = Turn(prompt="q1")
        turn.rounds.append(Round.tool("read_file", {"path": "/x"}, output={"ok": True}))
        assert turn.final_reply == ""


class TestSession:
    def _session(self, session_id: str = "s1", agent: str = "Iris") -> Session:
        return Session(SessionMetadata(session_id=session_id, agent=agent))

    def test_append_preserves_order(self):
        session = self._session()
        session.add_user("first")
        session.add_assistant("reply")
        assert [m.text for m in session.messages] == ["first", "reply"]

    def test_stores_metadata(self):
        session = self._session(session_id="s1", agent="Iris")
        assert session.metadata.session_id == "s1"
        assert session.metadata.agent == "Iris"

    def test_projection_closed_turn_is_coarse_summary(self):
        """A closed Turn projects as (prompt, final reply) only — its
        internal Rounds (system/synthetic/tool) are not shown to the model."""
        session = self._session()
        session.add_user("q1")
        session.add_system("context")
        session.add_assistant("a1")
        session.close_current_turn()
        msgs = session.projection()
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "q1"
        assert msgs[1].content == "a1"

    def test_projection_in_flight_turn_is_full_rounds(self):
        """The in-flight Turn (no final reply yet) projects its full Rounds
        so multi-round tool use stays coherent."""
        session = self._session()
        session.add_user("q1")
        session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "..."})
        # No assistant text yet — the Turn is still in flight.
        msgs = session.projection()
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert "[tool_call] read_file" in msgs[1].content
        assert msgs[1].role == "assistant"

    def test_projection_mixes_closed_and_in_flight(self):
        """Prior closed Turns are coarse; the current in-flight Turn is
        detailed — the structural rule."""
        session = self._session()
        # Closed turn 1: prompt + final reply only.
        session.add_user("q1")
        session.add_assistant("a1")
        # In-flight turn 2: full rounds visible.
        session.add_user("q2")
        session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "data"})
        msgs = session.projection()
        assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert msgs[0].content == "q1"
        assert msgs[1].content == "a1"
        assert msgs[2].content == "q2"
        assert "[tool_call] read_file" in msgs[3].content

    def test_projection_closed_turn_rounds_not_shown(self):
        """A closed Turn's tool/synthetic Rounds are retained on the Session
        but never shown to the model."""
        session = self._session()
        session.add_user("q1")
        session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "data"})
        session.add_synthetic("subagent result")
        session.add_assistant("concluded")
        session.close_current_turn()
        msgs = session.projection()
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "q1"
        assert msgs[1].content == "concluded"
        assert "read_file" not in " ".join(m.content for m in msgs)

    def test_projection_resumed_session_is_all_coarse(self):
        """A Session loaded from the archive has all Turns closed, so it
        projects coarse until the next Turn is opened — load-path independent."""
        session = self._session()
        session.add_user("q1")
        session.add_tool("read_file", params={"path": "/tmp/x"}, output={"content": "data"})
        session.add_assistant("a1")
        session.add_user("q2")
        session.add_assistant("a2")
        # Both turns closed (each has a final reply).
        msgs = session.projection()
        assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert [m.content for m in msgs] == ["q1", "a1", "q2", "a2"]
        assert "read_file" not in " ".join(m.content for m in msgs)

    def test_projection_in_flight_depends_on_state_not_load_path(self):
        """The same Turns project the same way regardless of how the Session
        was assembled — projection is a pure function of Turn state."""
        a = self._session()
        a.add_user("q1")
        a.add_assistant("a1")
        a.add_user("q2")
        a.add_tool("read_file", params={"path": "/x"}, output={"ok": True})

        b = self._session()
        b.add_user("q1")
        b.add_assistant("a1")
        b.add_user("q2")
        b.add_tool("read_file", params={"path": "/x"}, output={"ok": True})

        assert [m.content for m in a.projection()] == [m.content for m in b.projection()]

    def test_closed_turn_without_reply_projects_coarse(self):
        """A Turn closed with no text reply (e.g. MAX_ROUNDS exhausted mid
        tool-loop) projects coarse and is not treated as in-flight forever."""
        session = self._session()
        session.add_user("q1")
        session.add_tool("read_file", params={"path": "/x"}, output={"ok": True})
        session.close_current_turn()  # run ended without a text reply

        msgs = session.projection()
        # Prompt shown; no assistant reply to show; tool Rounds hidden.
        assert [m.role for m in msgs] == ["user"]
        assert msgs[0].content == "q1"

        # A later turn does not keep this one detailed.
        session.add_user("q2")
        session.add_assistant("a2")
        later = session.projection()
        assert [m.content for m in later] == ["q1", "q2", "a2"]

    def test_len_and_last(self):
        session = self._session()
        assert len(session) == 0
        assert session.last() is None
        session.add_user("hi")
        assert len(session) == 1
        assert session.last().text == "hi"

    def test_each_user_starts_a_turn(self):
        session = self._session()
        session.add_user("first")
        session.add_assistant("reply 1")
        session.add_user("second")
        session.add_assistant("reply 2")
        assert len(session.turns) == 2
        assert session.turns[0].summary == ("first", "reply 1")
        assert session.turns[1].summary == ("second", "reply 2")
        assert session.metadata.turn_count == 2
