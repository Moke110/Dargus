"""DargusRuntime — the single process-lifetime owner of runtime singletons.

v1.0.0 (design/2_runtime_structure.md): one runtime owns configuration, the single
reasoning LLM, tool/skill registries, D-Base tools, the
HookRegistry, the AgentFactory, a session-scoped ToolCache, the health
flag, and mode configuration. All Agents receive their dependencies from
the runtime, either directly or through the AgentFactory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dargus.runtime.modespec import ModeSpec, default_mode_config

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
    workspace_guard: Any | None = None  # WorkspaceGuard
    healthy: bool = True
    unhealthy_reason: str | None = None
    # ── Mode system (ADR-0002) ─────────────────────────────────────────
    mode: str = "auto"
    mode_config: dict[str, ModeSpec] = field(default_factory=dict)

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
        from dargus.tools.switch_mode import make_switch_mode_tool

        if self.agent_factory is None:
            self.agent_factory = AgentFactory(self)
        if self.tool_cache is None:
            self.tool_cache = ToolCache()
        if self.workspace_guard is None:
            self.workspace_guard = WorkspaceGuard(root=self.config.get("workspace_root"))
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()

        # Register general-purpose file Tools wired to the WorkspaceGuard.
        # These are not in registry.yaml because they require the runtime's
        # guard instance; they are registered programmatically here so every
        # Agent that receives this ToolRegistry sees them.
        self.tool_registry.register(make_read_file_tool(self.workspace_guard))
        self.tool_registry.register(make_write_file_tool(self.workspace_guard))

        # Register the switch_mode tool — always available regardless of mode.
        self.tool_registry.register(make_switch_mode_tool(self))

        # Populate mode_config from config or defaults.
        if not self.mode_config:
            self.mode_config = _mode_config_from_config(self.config)

    # ── Mode system (ADR-0002) ─────────────────────────────────────────

    def switch_mode(self, target: str) -> bool:
        """Transition to *target* mode, firing on_enter/on_exit hooks.

        Args:
            target: The target mode name. Must exist in ``mode_config``.

        Returns:
            True if the transition succeeded. Unknown modes are a logged
            no-op that returns False.
        """
        if target not in self.mode_config:
            logger.warning("switch_mode: unknown mode %r — no-op", target)
            return False

        current_spec = self.mode_config.get(self.mode)
        target_spec = self.mode_config[target]

        # Fire on_exit hook of current mode
        if current_spec and current_spec.on_exit and self.hook_registry:
            _run_named_hook(self.hook_registry, current_spec.on_exit, self, self.mode, target)

        # Fire on_enter hook of target mode
        if target_spec.on_enter and self.hook_registry:
            _run_named_hook(self.hook_registry, target_spec.on_enter, self, self.mode, target)

        old_mode = self.mode
        self.mode = target
        logger.info("Mode transition: %s → %s", old_mode, target)
        return True

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mode_config_from_config(config: dict) -> dict[str, ModeSpec]:
    """Build ``mode_config`` from the YAML ``modes:`` block or defaults.

    YAML entries provide tool/skill/hook lists and optional on_enter/on_exit
    hook names. System prompts default from :mod:`dargus.runtime.modespec`
    constants when absent from config.
    """
    defaults = default_mode_config()
    yaml_modes: dict = config.get("modes") or {}
    if not yaml_modes:
        return defaults

    result: dict[str, ModeSpec] = {}
    for mode_name in ("auto", "ingest", "predict"):
        yaml_entry = yaml_modes.get(mode_name, {}) if isinstance(yaml_modes, dict) else {}
        default_spec = defaults.get(mode_name, ModeSpec())
        result[mode_name] = ModeSpec(
            tools=yaml_entry.get("tools", default_spec.tools),
            skills=yaml_entry.get("skills", default_spec.skills),
            hooks=yaml_entry.get("hooks", default_spec.hooks),
            system_prompt=yaml_entry.get("system_prompt", default_spec.system_prompt),
            on_enter=yaml_entry.get("on_enter", default_spec.on_enter),
            on_exit=yaml_entry.get("on_exit", default_spec.on_exit),
        )
    return result


def _run_named_hook(
    hook_registry: Any,
    hook_name: str,
    runtime: DargusRuntime,
    from_mode: str,
    to_mode: str,
) -> None:
    """Run a named hook from the hook_registry if found.

    Named hooks are looked up by their ``__name__`` attribute. If no hook
    matches, the call is a silent no-op.
    """
    for hook in hook_registry.list_hooks():
        hook_id = getattr(hook, "__name__", None) or type(hook).__name__
        if hook_id == hook_name:
            try:
                from dargus.runtime.hooks import HookContext

                ctx = HookContext(
                    runtime=runtime,
                    task_spec={},
                    extra={"from_mode": from_mode, "to_mode": to_mode},
                )
                hook(ctx)
            except Exception:
                logger.exception(
                    "Hook %r failed during mode transition %s→%s",
                    hook_name,
                    from_mode,
                    to_mode,
                )
            return
    logger.debug("Named hook %r not found in registry", hook_name)
