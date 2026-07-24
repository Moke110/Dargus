"""AgentFactory — create Agent instances with dependency injection from RuntimeContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dargus.agents.base import BaseAgent
    from dargus.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class AgentFactory:
    """Creates Agent instances, wiring shared resources from RuntimeContext.

    In Phase A this is a skeleton — most factory methods are stubs that raise
    NotImplementedError.  Only ``base_agent()`` is functional, and even that
    falls back gracefully if BaseAgent does not yet accept DI kwargs (it will
    in Phase D).
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        self._runtime = runtime

    def base_agent(self, name: str) -> BaseAgent:
        """Create a BaseAgent with dependencies injected from the runtime.

        Tries to pass ``reasoning_llm``, ``tool_registry``, ``skill_registry``,
        ``knowledge_retrievers``, and ``hook_registry`` as keyword arguments.
        If BaseAgent's constructor does not accept these yet (pre-Phase D),
        falls back to ``BaseAgent(name=name)``.
        """
        from dargus.agents.base import BaseAgent

        try:
            return BaseAgent(
                name=name,
                config=self._runtime.config,
                reasoning_llm=self._runtime.reasoning_llm,
                tool_registry=self._runtime.tool_registry,
                skill_registry=self._runtime.skill_registry,
                knowledge_retrievers=self._runtime.knowledge_retrievers,
                hook_registry=self._runtime.hook_registry,
            )
        except TypeError as exc:
            logger.debug("BaseAgent does not accept DI kwargs yet — falling back: %s", exc)
            agent = BaseAgent(config=self._runtime.config)
            agent.name = name
            return agent

    def expert(self, domain: str):
        """Stub — create a DomainExpert for the given domain."""
        raise NotImplementedError("Expert factory not implemented yet")

    def d4_expert(self):
        """Stub — create a D4Expert coordinator."""
        raise NotImplementedError("Expert factory not implemented yet")

    def iris(self):
        """Stub — create an Iris commander Agent."""
        raise NotImplementedError("Expert factory not implemented yet")
