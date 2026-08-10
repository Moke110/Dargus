"""DargusRuntime — the single process-lifetime owner of runtime singletons.

One runtime owns configuration, the single reasoning LLM, the embedding
model (shared with D-Base), tool/skill registries, the AgentFactory, a
session-scoped ToolCache, and the health flag. All Agents receive their
dependencies from the runtime, either directly or through the AgentFactory.
The runtime holds **no per-session state** and at most one live Iris.
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
    agent_factory: Any | None = None
    tool_cache: Any | None = None
    workspace_guard: Any | None = None  # WorkspaceGuard
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
        from dargus.runtime.workspace import WorkspaceGuard
        from dargus.tools.cache import ToolCache
        from dargus.tools.file import make_read_file_tool, make_write_file_tool
        from dargus.tools.registry import ToolRegistry

        if self.agent_factory is None:
            self.agent_factory = AgentFactory(self)
        if self.tool_cache is None:
            self.tool_cache = ToolCache()
        if self.workspace_guard is None:
            self.workspace_guard = WorkspaceGuard(root=self.config.get("workspace_root"))
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()

        # Register general-purpose file Tools wired to the WorkspaceGuard.
        # These require the runtime's guard instance, so they are registered
        # programmatically here rather than as static definitions; every Agent
        # that receives this ToolRegistry sees them.
        self.tool_registry.register(make_read_file_tool(self.workspace_guard))
        self.tool_registry.register(make_write_file_tool(self.workspace_guard))

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

    @property
    def workspace_root(self) -> str:
        """The canonical workspace root off the WorkspaceGuard (or empty)."""
        guard = self.workspace_guard
        if guard is None:
            return ""
        return str(getattr(guard, "root", "") or "")


def health_check(runtime: DargusRuntime) -> bool:
    """Return True if both reasoning_llm and embedding_model are present."""
    return runtime.reasoning_llm is not None and runtime.embedding_model is not None
