"""Tests for the Conversation aggregate and ConvMessage model (T1 / #84).

Exercises the external behavior of the model: appending preserves order,
``to_llm_messages()`` projects each role correctly, and an assistant
Message records at most one tool record.
"""

from __future__ import annotations

from dargus.models.conversation import (
    Conversation,
    ConvMessage,
    ToolCall,
    ToolResult,
)


class TestConvMessage:
    def test_user_constructor_sets_role_and_text(self):
        msg = ConvMessage.user("hello")
        assert msg.role == "user"
        assert msg.text == "hello"

    def test_assistant_constructor_defaults_no_tool(self):
        msg = ConvMessage.assistant("done")
        assert msg.role == "assistant"
        assert msg.tool_call is None
        assert msg.tool_result is None

    def test_synthetic_constructor(self):
        msg = ConvMessage.synthetic("subagent result")
        assert msg.role == "synthetic"
        assert msg.text == "subagent result"

    def test_assistant_at_most_one_tool_record(self):
        """An assistant Message carries one tool call + one result pair."""
        msg = ConvMessage(
            role="assistant",
            tool_call=ToolCall("read_file", {"path": "/tmp/x"}),
            tool_result=ToolResult(output={"ok": True}),
        )
        assert msg.tool_call is not None
        assert msg.tool_result is not None
        assert msg.role == "assistant"

    def test_projects_user_role(self):
        msg = ConvMessage.user("hi")
        llm = msg.as_llm_message()
        assert llm.role == "user"
        assert llm.content == "hi"

    def test_projects_synthetic_role(self):
        msg = ConvMessage.synthetic("delegation")
        llm = msg.as_llm_message()
        assert llm.role == "synthetic"
        assert llm.content == "delegation"

    def test_projects_assistant_with_tool_renders_round(self):
        msg = ConvMessage(
            role="assistant",
            tool_call=ToolCall("read_file", {"path": "/tmp/x"}),
            tool_result=ToolResult(output={"content": "data"}),
        )
        llm = msg.as_llm_message()
        assert llm.role == "assistant"
        assert "[tool_call] read_file" in llm.content
        assert "path" in llm.content
        assert "data" in llm.content

    def test_projects_assistant_with_tool_error(self):
        msg = ConvMessage(
            role="assistant",
            tool_call=ToolCall("read_file", {}),
            tool_result=ToolResult(error="boom"),
        )
        llm = msg.as_llm_message()
        assert "[tool_error] boom" in llm.content


class TestConversation:
    def test_append_preserves_order(self):
        conv = Conversation(session_id="s1", agent="Iris")
        conv.add_user("first")
        conv.add_user("second")
        conv.add_assistant("reply")
        assert [m.text for m in conv.messages] == ["first", "second", "reply"]

    def test_stores_parent_id(self):
        conv = Conversation(session_id="child", agent="ClinicExpert", parent_id="parent")
        assert conv.parent_id == "parent"
        assert conv.session_id == "child"

    def test_to_llm_messages_projects_roles_in_order(self):
        conv = Conversation(session_id="s1", agent="Iris")
        conv.add_user("q1")
        conv.add_assistant("a1")
        conv.add_system("context")
        conv.add_synthetic("subagent done")
        msgs = conv.to_llm_messages()
        assert [m.role for m in msgs] == ["user", "assistant", "system", "synthetic"]
        assert msgs[0].content == "q1"
        assert msgs[1].content == "a1"

    def test_tool_message_appears_in_projection(self):
        conv = Conversation(session_id="s1", agent="Iris")
        conv.add_user("assess")
        conv.add_tool(
            ToolCall("read_file", {"path": "/tmp/x"}),
            ToolResult(output={"content": "..."}),
        )
        conv.add_assistant("concluded")
        msgs = conv.to_llm_messages()
        assert len(msgs) == 3
        assert "[tool_call] read_file" in msgs[1].content
        assert msgs[1].role == "assistant"

    def test_len_and_last(self):
        conv = Conversation(session_id="s1", agent="Iris")
        assert len(conv) == 0
        assert conv.last() is None
        conv.add_user("hi")
        assert len(conv) == 1
        assert conv.last().text == "hi"
