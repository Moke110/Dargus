"""DargusRuntime — the single process-lifetime owner of runtime singletons.

v1.0.0 (design/2_runtime_structure.md): one runtime owns configuration, the single
reasoning LLM, tool/skill registries, D-Base tools, the
HookRegistry, the AgentFactory, a session-scoped ToolCache, and the health
flag. All Agents receive their dependencies from the runtime, either
directly or through the AgentFactory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.models.embedding import EmbeddingModel
    from dargus.models.reasoning import ReasoningLLM

logger = logging.getLogger(__name__)


@dataclass
class DargusRuntime:
    """Holds all shared, pre-warmed resources for a single process/session.

    Fields marked as ``Any`` are forward references to types concretised in
    later phases (SkillRegistry, DBaseStore).
    """

    config: dict[str, Any] = field(default_factory=dict)
    reasoning_llm: ReasoningLLM | None = None
    embedding_model: EmbeddingModel | None = None
    tool_registry: Any | None = None
    skill_registry: Any | None = None
    dbase_store: Any | None = None
    hook_registry: Any | None = None
    agent_factory: Any | None = None
    tool_cache: Any | None = None
    healthy: bool = True
    unhealthy_reason: str | None = None

    def mark_unhealthy(self, reason: str) -> None:
        """Flip the health flag after an unrecoverable dependency failure.

        The runtime starts healthy; only an unrecoverable failure (D-Base
        inaccessible, model unavailable) turns it unhealthy. Entry points
        refuse new sessions while unhealthy; recovery requires a restart.
        """
        self.healthy = False
        self.unhealthy_reason = reason
        logger.error("DargusRuntime marked unhealthy: %s", reason)

    def __post_init__(self) -> None:
        from dargus.runtime.factory import AgentFactory
        from dargus.tools.cache import ToolCache

        if self.agent_factory is None:
            self.agent_factory = AgentFactory(self)
        if self.tool_cache is None:
            self.tool_cache = ToolCache()

    def ensure_healthy(self) -> None:
        """Raise if the runtime is unhealthy — called by API entry points."""
        if not self.healthy:
            reason = self.unhealthy_reason or "unknown"
            raise RuntimeError(
                f"DargusRuntime is unhealthy ({reason}) — refusing new session; "
                "restart the runtime to recover"
            )

    def shutdown(self) -> None:
        """Release session-scoped resources (ToolCache) and mark unhealthy."""
        if self.tool_cache is not None:
            self.tool_cache.close()
        self.healthy = False


def health_check(runtime: DargusRuntime) -> bool:
    """Return True if both reasoning_llm and embedding_model are present."""
    return runtime.reasoning_llm is not None and runtime.embedding_model is not None
