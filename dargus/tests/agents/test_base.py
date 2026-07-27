"""Test Phase D: BaseAgent dependency injection and hook triggering."""

from unittest.mock import MagicMock

from dargus.agents.base import BaseAgent
from dargus.agents.skill_registry import SkillRegistry
from dargus.runtime.hooks import HookContext, HookPoint, HookRegistry
from dargus.tools.registry import ToolRegistry


class _MinimalAgent(BaseAgent):
    """Concrete subclass for testing — no abstract methods."""

    name = "_MinimalAgent"
    PERMITTED_TOOLS = []
    PERMITTED_KNOWLEDGE = []
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


def test_base_agent_injected_knowledge_retrievers():
    """Injected knowledge retrievers are used."""
    mock_retriever = MagicMock()
    agent = _MinimalAgent(
        name="test",
        knowledge_retrievers={"test_source": mock_retriever},
    )
    assert agent._knowledge_retrievers["test_source"] is mock_retriever


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
