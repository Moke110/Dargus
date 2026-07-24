"""RuntimeContext — shared per-process resources for the Dargus runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dargus.models.embedding import EmbeddingModel
    from dargus.models.reasoning import ReasoningLLM


@dataclass
class RuntimeContext:
    """Holds all shared, pre-warmed resources for a single process/session.

    Fields marked as ``Any`` are forward references to types that will be
    concretised in later phases (ToolRegistry, SkillRegistry, DBaseManager,
    HookRegistry).
    """

    config: dict[str, Any] = field(default_factory=dict)
    reasoning_llm: ReasoningLLM | None = None
    embedding_model: EmbeddingModel | None = None
    tool_registry: Any | None = None
    skill_registry: Any | None = None
    knowledge_retrievers: dict[str, Any] = field(default_factory=dict)
    dbase_manager: Any | None = None
    hook_registry: Any | None = None
    healthy: bool = False


def health_check(ctx: RuntimeContext) -> bool:
    """Perform a basic health check on a RuntimeContext.

    Returns True if both reasoning_llm and embedding_model are present (not None).
    """
    return ctx.reasoning_llm is not None and ctx.embedding_model is not None
