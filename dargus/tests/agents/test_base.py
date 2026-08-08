"""Test Phase D: BaseAgent dependency injection and hook triggering."""

from typing import Any
from unittest.mock import MagicMock

from dargus.agents.base import BaseAgent
from dargus.agents.skill_registry import SkillRegistry
from dargus.runtime.hooks import HookContext, HookPoint, HookRegistry
from dargus.tools.base import Tool
from dargus.tools.registry import ToolRegistry


class _MinimalAgent(BaseAgent):
    """Concrete subclass for testing — no abstract methods."""

    name = "_MinimalAgent"
    PERMITTED_TOOLS = []
    SUPPORTED_SKILLS = []


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


def test_base_agent_no_di_works():
    """BaseAgent() with just a name or config must work."""
    agent = _MinimalAgent(name="test")
    assert agent.name == "test"
    assert agent.config is not None
    assert agent._tool_registry is not None
    assert agent._skill_registry is not None


# ------------------------------------------------------------------
# Conversation as single source of truth (T2 / #85)
# ------------------------------------------------------------------


class _StubLLM:
    """Scripted reasoning LLM: returns each queued response in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list] = []

    def chat(self, messages: list) -> Any:
        self.calls.append(list(messages))
        if self.responses:
            return type("R", (), {"content": self.responses.pop(0)})()
        return type("R", (), {"content": '{"mode": "auto", "action": "text", "text": "done"}'})()


def _make_agent(runtime=None, responses: list[str] | None = None):
    agent = _MinimalAgent(
        name="test",
        hook_registry=HookRegistry(),
        reasoning_llm=_StubLLM(responses or []),
    )
    if runtime is not None:
        agent._runtime = runtime
    return agent


def test_run_appends_one_assistant_message_per_round():
    """T2: a multi-round run (tool_call then text) appends one assistant
    Message per round to the Conversation."""
    agent = _make_agent(
        responses=[
            '{"mode": "auto", "action": "tool_call", "tool": "dbase_query", "params": {}}',
            '{"mode": "auto", "action": "text", "text": "concluded"}',
        ]
    )
    report = agent.run({"query": "assess"})
    assert report.rounds == 2

    conv = agent._resolve_conversation({"query": "assess"})
    # Sanity: the text message is the terminal round
    assert conv.last().text == "concluded"
    roles = [m.role for m in conv.messages]
    # user + tool-call assistant + text assistant
    assert roles == ["user", "assistant", "assistant"]
    tool_msgs = [m for m in conv.messages if m.tool_call is not None]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call.name == "dbase_query"


def test_run_settles_interrupted_tool_as_error_message():
    """T2: a Tool that raises settles an error Message for that round."""
    registry = ToolRegistry()
    boom = Tool(
        name="boom_tool",
        description="always fails",
        parameters=[],
        output={},
    )
    boom.bind(lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    registry.register(boom)

    agent = _MinimalAgent(name="test", tool_registry=registry)
    agent._mode_config = {
        "auto": type("MS", (), {"system_prompt": "", "tools": ["boom_tool"], "skills": []})(),
    }
    agent._reasoning_llm = _StubLLM(
        ['{"mode": "auto", "action": "tool_call", "tool": "boom_tool", "params": {}}']
    )
    agent._hook_registry = HookRegistry()
    agent.run({"query": "x"})

    conv = agent._resolve_conversation({"query": "x"})
    tool_msgs = [m for m in conv.messages if m.tool_call is not None]
    assert len(tool_msgs) == 1
    assert "kaboom" in (tool_msgs[0].tool_result.error or "")


def test_run_messages_carry_active_mode():
    """T2: Messages carry the Mode active when they were produced."""
    agent = _make_agent(responses=['{"mode": "predict", "action": "text", "text": "ok"}'])
    agent._mode = "predict"
    agent.run({"query": "x"})

    conv = agent._resolve_conversation({"query": "x"})
    assert all(m.mode == "predict" for m in conv.messages)


class _RecordingLLM:
    """Scripted LLM that records the messages it receives at each call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list] = []

    def chat(self, messages: list) -> Any:
        self.calls.append(list(messages))
        if self.responses:
            return type("R", (), {"content": self.responses.pop(0)})()
        return type("R", (), {"content": '{"mode": "auto", "action": "text", "text": "done"}'})()


def test_model_visible_context_is_projected_messages():
    """#94 (SPEC-A): chat() receives the projected Conversation messages as
    role/content, not a re-serialized JSON blob in one user_prompt.

    A tool-call round renders as an assistant message carrying the tool
    content, and the terminal text round is preserved verbatim.
    """
    from dargus.models.reasoning import Message

    agent = _MinimalAgent(
        name="test",
        hook_registry=HookRegistry(),
        reasoning_llm=_RecordingLLM(
            [
                '{"mode": "auto", "action": "tool_call", "tool": "dbase_query", "params": {}}',
                '{"mode": "auto", "action": "text", "text": "concluded"}',
            ]
        ),
    )
    agent.run({"query": "assess"})

    assert len(agent._reasoning_llm.calls) == 2
    for call in agent._reasoning_llm.calls:
        # First element is the system+framing message; the dialogue follows.
        assert call[0].role == "system"
        assert "# Task framing" in call[0].content
        assert all(isinstance(m, Message) for m in call)

    # The projection at each call is the Conversation as it stood *before*
    # that round's own LLM call. Round 1: system + the opening user message.
    first, second = agent._reasoning_llm.calls
    assert [m.role for m in first] == ["system", "user"]

    # Round 2: the round-1 tool call now renders as an assistant message
    # carrying the tool content SPEC-A maps (not a JSON blob).
    assert [m.role for m in second] == ["system", "user", "assistant"]
    assert "[tool_call] dbase_query" in second[-1].content


def test_two_user_turns_accumulate_in_one_conversation():
    """T4: two ask()-style turns on one reused runtime append user messages
    in order and the second turn sees the first turn's messages."""
    from dargus.runtime.context import DargusRuntime

    runtime = DargusRuntime()
    # Use a stub reasoning LLM wired via DI.
    agent = _MinimalAgent(
        name="Iris",
        hook_registry=HookRegistry(),
        reasoning_llm=_StubLLM(
            [
                '{"mode": "auto", "action": "text", "text": "first reply"}',
                '{"mode": "auto", "action": "text", "text": "second reply"}',
            ]
        ),
    )
    agent._runtime = runtime

    agent.run({"query": "what is aspirin?", "session_id": "dialogue", "_user_turn": True})
    agent.run({"query": "and metformin?", "session_id": "dialogue", "_user_turn": True})

    conv = runtime.get_conversation("dialogue", "Iris")
    texts = [m.text for m in conv.messages]
    # Two user turns + two assistant replies, in order.
    n_aspirin = sum(1 for t in texts if "what is aspirin?" in t)
    n_metformin = sum(1 for t in texts if "and metformin?" in t)
    assert n_aspirin == 1
    assert n_metformin == 1
    assert "first reply" in texts
    assert "second reply" in texts
    assert len(conv.messages) == 4


def test_distinct_sessions_get_distinct_conversations():
    """T4 regression: a reused agent resolves a separate Conversation per
    session_id instead of caching the first session's log."""
    from dargus.runtime.context import DargusRuntime

    runtime = DargusRuntime()
    agent = _MinimalAgent(
        name="Iris",
        hook_registry=HookRegistry(),
        reasoning_llm=_StubLLM(
            [
                '{"mode": "auto", "action": "text", "text": "dialogue reply"}',
                '{"mode": "auto", "action": "text", "text": "predict reply"}',
            ]
        ),
    )
    agent._runtime = runtime

    agent.run({"query": "follow up", "session_id": "dialogue", "_user_turn": True})
    agent.run({"query": "predict", "session_id": "predict:d1:d2:IC50", "_user_turn": True})

    dialogue = runtime.get_conversation("dialogue", "Iris")
    predict = runtime.get_conversation("predict:d1:d2:IC50", "Iris")
    # Distinct logs: the predict turn did not bleed into the dialogue log.
    assert dialogue is not predict
    assert len(dialogue.messages) == 2
    assert len(predict.messages) == 2
    assert "follow up" in dialogue.messages[0].text
    assert "predict" in predict.messages[0].text


def test_base_agent_config_only_works():
    """BaseAgent(config=...) must work without DI."""
    agent = _MinimalAgent(config={"projects": {"root_dir": "/tmp"}})
    assert agent.config["projects"]["root_dir"] == "/tmp"


def test_base_agent_legacy_positional_config():
    """BaseAgent({...}) — old-style positional dict as config."""
    agent = _MinimalAgent({"projects": {"root_dir": "/tmp"}})
    assert agent.config["projects"]["root_dir"] == "/tmp"


# ------------------------------------------------------------------
# Dependency injection
# ------------------------------------------------------------------


def test_base_agent_injected_tool_registry():
    """Injected ToolRegistry is used instead of creating a default."""
    custom = ToolRegistry()
    agent = _MinimalAgent(name="test", tool_registry=custom)
    assert agent._tool_registry is custom


def test_base_agent_injected_skill_registry():
    """Injected SkillRegistry is used instead of creating a default."""
    custom = SkillRegistry()
    agent = _MinimalAgent(name="test", skill_registry=custom)
    assert agent._skill_registry is custom


def test_base_agent_injected_hook_registry():
    """Injected HookRegistry is stored."""
    custom = HookRegistry()
    agent = _MinimalAgent(name="test", hook_registry=custom)
    assert agent._hook_registry is custom


def test_base_agent_none_hook_registry():
    """When hook_registry is None (default), no hooks run — no crash."""
    agent = _MinimalAgent(name="test")
    assert agent._hook_registry is None


# ------------------------------------------------------------------
# Hook triggering in run()
# ------------------------------------------------------------------


def test_run_triggers_reason_end_hook():
    """When hook_registry is injected, REASON_END fires after reasoning."""
    registry = HookRegistry()
    hook_tracker = MagicMock()

    def track_reason_end(ctx: HookContext) -> HookContext:
        hook_tracker(ctx)
        return ctx

    registry.register(HookPoint.REASON_END, track_reason_end)

    agent = _MinimalAgent(name="test", hook_registry=registry)
    report = agent.run({"goal": "simple_test"})

    assert report is not None
    assert hook_tracker.call_count >= 1
    call_ctx = hook_tracker.call_args[0][0]
    assert isinstance(call_ctx, HookContext)
    assert call_ctx.agent is agent


def test_run_triggers_act_end_hook():
    """When hook_registry is injected, ACT_END fires after execution."""
    registry = HookRegistry()
    hook_tracker = MagicMock()

    def track_act_end(ctx: HookContext) -> HookContext:
        hook_tracker(ctx)
        return ctx

    registry.register(HookPoint.ACT_END, track_act_end)

    agent = _MinimalAgent(name="test", hook_registry=registry)
    report = agent.run({"goal": "simple_test"})

    assert report is not None
    assert hook_tracker.call_count >= 1


def test_run_triggers_perceive_end_hook():
    """When hook_registry is injected, PERCEIVE_END fires after perceiving."""
    registry = HookRegistry()
    hook_tracker = MagicMock()

    def track_perceive_end(ctx: HookContext) -> HookContext:
        hook_tracker(ctx)
        return ctx

    registry.register(HookPoint.PERCEIVE_END, track_perceive_end)

    agent = _MinimalAgent(name="test", hook_registry=registry)
    report = agent.run({"goal": "simple_test"})

    assert report is not None
    assert hook_tracker.call_count >= 1


def test_run_does_not_crash_without_hook_registry():
    """BaseAgent.run() is safe when hook_registry is None."""
    agent = _MinimalAgent(name="test")
    report = agent.run({"goal": "simple_test"})
    assert report is not None
    assert report.agent_name == "test"  # DI constructor sets self.name
