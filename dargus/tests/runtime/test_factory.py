"""Tests for AgentFactory."""

import pytest

from dargus.agents.base import BaseAgent
from dargus.runtime.context import RuntimeContext
from dargus.runtime.factory import AgentFactory


class TestAgentFactory:
    """Tests for AgentFactory."""

    def test_constructor_stores_runtime(self):
        ctx = RuntimeContext()
        factory = AgentFactory(ctx)
        assert factory._runtime is ctx

    def test_base_agent_creates_instance(self):
        """base_agent() creates a BaseAgent (with DI fallback)."""
        ctx = RuntimeContext()
        factory = AgentFactory(ctx)
        agent = factory.base_agent("TestAgent")

        assert isinstance(agent, BaseAgent)
        assert agent.name == "TestAgent"

    def test_base_agent_with_di_fallback(self):
        """When BaseAgent doesn't accept DI kwargs, fallback works gracefully."""
        ctx = RuntimeContext()
        ctx.reasoning_llm = object()  # type: ignore[assignment]
        factory = AgentFactory(ctx)
        agent = factory.base_agent("DI_Agent")

        assert isinstance(agent, BaseAgent)
        assert agent.name == "DI_Agent"

    def test_expert_raises_not_implemented(self):
        factory = AgentFactory(RuntimeContext())
        with pytest.raises(NotImplementedError, match="Expert factory not implemented yet"):
            factory.expert("cardiology")

    def test_d4_expert_raises_not_implemented(self):
        factory = AgentFactory(RuntimeContext())
        with pytest.raises(NotImplementedError, match="Expert factory not implemented yet"):
            factory.d4_expert()

    def test_iris_raises_not_implemented(self):
        factory = AgentFactory(RuntimeContext())
        with pytest.raises(NotImplementedError, match="Expert factory not implemented yet"):
            factory.iris()
