"""Test Phase D: BaseAgent PRA loop and Session integration."""

from typing import Any

from dargus.agents.base import BaseAgent
from dargus.agents.skill_registry import SkillRegistry
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
# Session as single source of truth (T2 / #85)
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
        return type("R", (), {"content": '{"action": "text", "text": "done"}'})()


def _make_agent(runtime=None, responses: list[str] | None = None, name: str = "test"):
    agent = _MinimalAgent(
        name=name,
        reasoning_llm=_StubLLM(responses or []),
    )
    if runtime is not None:
        agent._runtime = runtime
    return agent


def test_run_appends_one_assistant_message_per_round():
    """T2: a multi-round run (tool_call then text) appends one assistant
    Round per round to the Session."""
    agent = _make_agent(
        responses=[
            '{"action": "tool_call", "tool": "read_file", "params": {}}',
            '{"action": "text", "text": "concluded"}',
        ]
    )
    report = agent.run({"query": "assess"})
    assert report.rounds == 2

    session = agent._resolve_session({"query": "assess"})
    # Sanity: the text message is the terminal round
    assert session.last().text == "concluded"
    roles = [m.role for m in session.messages]
    # user + tool-call assistant + text assistant
    assert roles == ["user", "assistant", "assistant"]
    tool_rounds = [m for m in session.messages if m.tool_name is not None]
    assert len(tool_rounds) == 1
    assert tool_rounds[0].tool_name == "read_file"


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

    agent = _MinimalAgent(
        name="test",
        tool_registry=registry,
        reasoning_llm=_StubLLM(['{"action": "tool_call", "tool": "boom_tool", "params": {}}']),
    )
    agent.PERMITTED_TOOLS = ["boom_tool"]
    agent.run({"query": "x"})

    session = agent._resolve_session({"query": "x"})
    tool_rounds = [m for m in session.messages if m.tool_name is not None]
    assert len(tool_rounds) == 1
    assert "kaboom" in (tool_rounds[0].tool_error or "")


def test_run_blocks_tool_not_in_permitted_tools():
    """ACT refuses a tool call outside the agent's PERMITTED_TOOLS."""
    agent = _MinimalAgent(
        name="test",
        reasoning_llm=_StubLLM(['{"action": "tool_call", "tool": "write_file", "params": {}}']),
    )
    agent.run({"query": "x"})

    session = agent._resolve_session({"query": "x"})
    tool_rounds = [m for m in session.messages if m.tool_name is not None]
    assert len(tool_rounds) == 1
    output = tool_rounds[0].tool_output
    assert isinstance(output, dict)
    inner = output.get("output", {})
    assert isinstance(inner, dict)
    assert "not permitted" in str(inner.get("error", ""))


class _RecordingLLM:
    """Scripted LLM that records the messages it receives at each call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list] = []

    def chat(self, messages: list) -> Any:
        self.calls.append(list(messages))
        if self.responses:
            return type("R", (), {"content": self.responses.pop(0)})()
        return type("R", (), {"content": '{"action": "text", "text": "done"}'})()


def test_model_visible_context_is_projected_messages():
    """#94 (SPEC-A): chat() receives the projected Session messages as
    role/content, not a re-serialized JSON blob in one user_prompt.

    A tool-call round renders as an assistant message carrying the tool
    content, and the terminal text round is preserved verbatim.
    """
    from dargus.models.reasoning import Message

    agent = _MinimalAgent(
        name="test",
        reasoning_llm=_RecordingLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "params": {}}',
                '{"action": "text", "text": "concluded"}',
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

    # The projection at each call is the Session as it stood *before*
    # that round's own LLM call. Round 1: system + the opening user message.
    first, second = agent._reasoning_llm.calls
    assert [m.role for m in first] == ["system", "user"]

    # Round 2: the round-1 tool call now renders as an assistant message
    # carrying the tool content SPEC-A maps (not a JSON blob).
    assert [m.role for m in second] == ["system", "user", "assistant"]
    assert "[tool_call] read_file" in second[-1].content


def test_two_user_turns_accumulate_in_one_session():
    """T4: two ask()-style turns on one reused Iris append user messages
    in order and the second turn sees the first turn's messages."""
    agent = _make_agent(
        name="Iris",
        responses=[
            '{"action": "text", "text": "first reply"}',
            '{"action": "text", "text": "second reply"}',
        ],
    )

    agent.run({"query": "what is aspirin?"})
    agent.run({"query": "and metformin?"})

    session = agent._resolve_session({})
    texts = [m.text for m in session.messages]
    # Two user turns + two assistant replies, in order.
    n_aspirin = sum(1 for t in texts if "what is aspirin?" in t)
    n_metformin = sum(1 for t in texts if "and metformin?" in t)
    assert n_aspirin == 1
    assert n_metformin == 1
    assert "first reply" in texts
    assert "second reply" in texts
    assert len(session.messages) == 4


def test_projection_across_turns_is_coarse_prior_plus_detailed_current():
    """#105: after several completed turns, the model-visible context is
    coarse for prior turns and full only for the in-flight turn."""
    from dargus.models.reasoning import Message

    agent = _make_agent(
        name="Iris",
        responses=[
            '{"action": "text", "text": "first reply"}',  # turn 1 closes
            '{"action": "tool_call", "tool": "read_file", "params": {}}',  # turn 2 round 1
            '{"action": "text", "text": "second reply"}',  # turn 2 closes
        ],
    )
    agent.run({"query": "turn one?"})
    agent.run({"query": "turn two?"})

    # Turn 1 is closed; turn 2's rounds completed, so the final Session is
    # all-coarse. The in-flight view is what the LLM saw during turn 2's
    # round 1 (before its final reply): prior turn coarse, current turn full.
    session = agent._resolve_session({})
    assert [m.role for m in session.projection()] == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in session.projection()] == [
        '{"query": "turn one?"}',
        "first reply",
        '{"query": "turn two?"}',
        "second reply",
    ]

    # The LLM's round-2 call in turn 2 saw turn 1 coarse + turn 2 in-flight
    # (the round-1 tool result in full, no final reply yet).
    second_turn_round_two = agent._reasoning_llm.calls[2]
    assert [m.role for m in second_turn_round_two] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert second_turn_round_two[1].content == '{"query": "turn one?"}'
    assert second_turn_round_two[2].content == "first reply"
    assert '"turn two?"' in second_turn_round_two[3].content
    assert "[tool_call] read_file" in second_turn_round_two[4].content
    assert all(isinstance(m, Message) for m in second_turn_round_two)


def test_distinct_agents_get_distinct_sessions():
    """T4 regression: each agent instance owns its own Session — no
    cross-agent bleed."""
    agent_a = _make_agent(
        name="Iris",
        responses=['{"action": "text", "text": "dialogue reply"}'],
    )
    agent_b = _make_agent(
        name="ClinicExpert",
        responses=['{"action": "text", "text": "predict reply"}'],
    )

    agent_a.run({"query": "follow up"})
    agent_b.run({"query": "predict"})

    session_a = agent_a._resolve_session({})
    session_b = agent_b._resolve_session({})
    # Distinct logs: the predict turn did not bleed into the dialogue log.
    assert session_a is not session_b
    assert len(session_a.messages) == 2
    assert len(session_b.messages) == 2
    assert "follow up" in session_a.messages[0].text
    assert "predict" in session_b.messages[0].text


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


def test_base_agent_default_system_prompt_is_empty():
    """The default agent system_prompt is empty; subclasses declare one."""
    agent = _MinimalAgent(name="test")
    assert agent.system_prompt == ""
