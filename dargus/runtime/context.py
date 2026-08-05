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
    # ── Conversations (ADR-0003 / SPEC-B) ───────────────────────────────
    # Every agent's Conversation, keyed by (session_id, agent). Owned by the
    # runtime so the log survives agent churn and API turns.
    conversation_store: dict[tuple[str, str], Any] = field(default_factory=dict)
    # The durable session object hooks receive — survives rounds and turns.
    session: dict[str, Any] = field(default_factory=dict)
    # ── Subagent spawning (SPEC-C) ──────────────────────────────────────
    # Stack of active subagent sessions; empty while Iris runs (depth 0).
    # Non-empty means a Subagent is mid-run, so spawn_expert is denied
    # (depth-1 guard — only Iris may spawn).
    spawn_stack: list[str] = field(default_factory=list)
    # Session id of the agent currently running (set by BaseAgent.run) so the
    # spawn tool can link a Subagent's Conversation to its parent's.
    current_session_id: str | None = None

    def get_conversation(self, session_id: str, agent: str, parent_id: str | None = None) -> Any:
        """Return (creating if needed) the Conversation for session/agent.

        ``parent_id`` links a Subagent's Conversation to its parent's
        (opencode ``parentID`` analogue).
        """
        key = (session_id, agent)
        conv = self.conversation_store.get(key)
        if conv is None:
            from dargus.models.conversation import Conversation

            conv = Conversation(session_id=session_id, agent=agent, parent_id=parent_id)
            self.conversation_store[key] = conv
        return conv

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
        from dargus.runtime.hooks import HookPoint, HookRegistry, WorkspaceGuardHook
        from dargus.runtime.mode_tag import ModeTagValidationHook
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
        if self.hook_registry is None:
            self.hook_registry = HookRegistry()

        # Wire the agent-loop hooks every agent should see (ADR-0002).
        # Mode-tag validation runs at REASON_END to block off-mode ACT; the
        # workspace guard backstops path-typed tool params at ACT_START.
        self.hook_registry.register(HookPoint.REASON_END, ModeTagValidationHook())
        self.hook_registry.register(HookPoint.ACT_START, WorkspaceGuardHook())

        # Register general-purpose file Tools wired to the WorkspaceGuard.
        # These are not in registry.yaml because they require the runtime's
        # guard instance; they are registered programmatically here so every
        # Agent that receives this ToolRegistry sees them.
        self.tool_registry.register(make_read_file_tool(self.workspace_guard))
        self.tool_registry.register(make_write_file_tool(self.workspace_guard))

        # Register the switch_mode tool — always available regardless of mode.
        self.tool_registry.register(make_switch_mode_tool(self))

        # Bind real implementations for the dbase_* Tools so any Agent (Iris or
        # an Expert Subagent) can actually query/write D-Base (SPEC-C: Experts
        # self-serve evidence). The registry.yaml entries are schema-only stubs;
        # this wires them to the runtime's DBaseStore when one is available.
        self._bind_dbase_tools()

        # Register the spawn_expert tool (SPEC-C) — only available in predict
        # mode. It is bound lazily at first Iris construction because it needs
        # the AgentFactory + Iris back-references.
        self._spawn_tool = None

        # Populate mode_config from config or defaults.
        if not self.mode_config:
            self.mode_config = _mode_config_from_config(self.config)

    def _bind_dbase_tools(self) -> None:
        """Bind dbase_query/dbase_write/... with lazily-resolved store impls.

        The registry.yaml entries are schema-only stubs. This replaces them
        with implementations that resolve the D-Base store per call: the
        runtime's ``dbase_store`` when wired, else the global D-Base. The
        runtime's ``dbase_store`` field is left untouched by default
        construction (existing contract).
        """
        from dargus.tools.base import Tool, ToolParam
        from dargus.tools.dbase import (
            dbase_query,
            dbase_status,
            dbase_update_status,
            dbase_write,
            dbase_write_summary,
        )

        def _store() -> Any | None:
            if self.dbase_store is not None:
                return self.dbase_store
            try:
                from dargus.dbase import DBase
                from dargus.dbase.store import DBaseStore

                return DBaseStore(DBase.global_instance())
            except Exception:
                return None

        specs: dict[str, Tool] = {}

        q = Tool(
            name="dbase_query",
            description="Query the D-Base evidence store by drug/disease/level filters",
            parameters=[
                ToolParam("x_entity", "string", description="Intervention entity id (CURIE)"),
                ToolParam("disease_id", "string", description="Disease id (CURIE)"),
                ToolParam("y_type", "string", description="Readout/endpoint type"),
                ToolParam("y_category", "string", description="Readout category"),
                ToolParam("level", "string", description="Biological evidence level"),
                ToolParam("evidence_design", "string", description="Evidence design"),
                ToolParam("limit", "integer", default=100, description="Max records"),
            ],
            output={"type": "object", "properties": {"records": {"type": "array"}}},
        )
        q.bind(lambda **kw: dbase_query(_store(), kw))
        specs["dbase_query"] = q

        w = Tool(
            name="dbase_write",
            description="Write an evidence record through the single-writer D-Base API",
            parameters=[ToolParam("record", "object", required=True, description="Evidence dict")],
            output={"type": "object", "properties": {"evidence_id": {"type": "string"}}},
        )
        w.bind(lambda record: dbase_write(_store(), record))
        specs["dbase_write"] = w

        s = Tool(
            name="dbase_status",
            description="Report D-Base state (record count, shards, parquet view)",
            parameters=[],
            output={"type": "object", "properties": {"record_count": {"type": "integer"}}},
        )
        s.bind(lambda: dbase_status(_store()))
        specs["dbase_status"] = s

        us = Tool(
            name="dbase_update_status",
            description="Append a lifecycle status transition (supersede, retract, holdout)",
            parameters=[
                ToolParam("evidence_id", "string", required=True),
                ToolParam(
                    "status",
                    "string",
                    required=True,
                    enum=["active", "superseded", "retracted", "holdout-test", "holdout-valid"],
                ),
                ToolParam("superseded_by", "string"),
            ],
            output={"type": "object", "properties": {"status": {"type": "string"}}},
        )
        us.bind(
            lambda evidence_id, status, superseded_by=None: dbase_update_status(
                _store(), evidence_id, status, superseded_by
            )
        )
        specs["dbase_update_status"] = us

        ws = Tool(
            name="dbase_write_summary",
            description="Write or replace the LLM summary sidecar entry for a record",
            parameters=[
                ToolParam("evidence_id", "string", required=True),
                ToolParam("summary", "string", required=True),
            ],
            output={"type": "object", "properties": {"written": {"type": "boolean"}}},
        )
        ws.bind(lambda evidence_id, summary: dbase_write_summary(_store(), evidence_id, summary))
        specs["dbase_write_summary"] = ws

        for name, tool in specs.items():
            if _store() is None:
                logger.debug("DargusRuntime: no D-Base available — %s stays unbound", name)
                continue
            self.tool_registry.register(tool)

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
