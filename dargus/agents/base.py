"""BaseAgent — Harness skeleton with Perceive -> Reason -> Act execution loop."""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC
from pathlib import Path
from typing import Any

import yaml

from dargus.agents.report import AgentReport, CallTrace
from dargus.agents.skill_registry import SkillRegistry
from dargus.models.conversation import Conversation, ToolCall, ToolResult
from dargus.models.reasoning import Message, ReasoningLLM
from dargus.runtime.hooks import HookContext, HookPoint, HookRegistry
from dargus.tools.base import Tool
from dargus.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Harness-equipped agent base class.

    Subclasses declare:
      - PERMITTED_TOOLS: list[str]
      - SUPPORTED_SKILLS: list[str]
      - SUPPORTED_LEVELS: tuple[str, ...]
    """

    name: str = "BaseAgent"

    # --- subclass overrides ---
    PERMITTED_TOOLS: list[str] = []
    SUPPORTED_SKILLS: list[str] = []
    SUPPORTED_LEVELS: tuple[str, ...] = ()
    MAX_ROUNDS: int = 5

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        hook_registry: HookRegistry | None = None,
        mode: str = "auto",
        mode_config: dict[str, Any] | None = None,
        runtime: Any | None = None,
    ):
        # Backward compat: if first positional arg is a dict, treat as config not name
        if isinstance(name, dict):
            config = name if config is None else config
            name = None

        if name is not None:
            self.name = name
        self.config = config or self._load_default_config()
        self.skills: set[str] = set()

        # DI or defaults — each injected value takes priority; None means "create default"
        self._tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._skill_registry = skill_registry if skill_registry is not None else SkillRegistry()
        self._hook_registry: HookRegistry | None = hook_registry

        # Reasoning LLM comes only from DI (the runtime's single model);
        # without it the agent runs in deterministic stub mode.
        self._reasoning_llm: ReasoningLLM | None = reasoning_llm

        # Mode system (ADR-0002)
        self._mode: str = mode
        self._mode_config: dict[str, Any] = mode_config or {}

        # Back-reference to the owning DargusRuntime, injected by AgentFactory.
        # Runtime-dependent hooks (mode-tag validation, workspace guard) read
        # state off this; None in standalone/test contexts, where those hooks
        # pass through.
        self._runtime: Any | None = runtime

        # The agent's Conversation (ADR-0003) — the single source of truth
        # for context. Resolved through the runtime's conversation store when
        # one is wired (T4); a per-instance fallback otherwise.
        self._conversation: Conversation | None = None

        self._validate_permissions()

    def _validate_permissions(self) -> None:
        """Ensure SUPPORTED_SKILLS' required_tools are subset of PERMITTED_TOOLS."""
        for skill_name in self.SUPPORTED_SKILLS:
            try:
                skill = self._skill_registry.get(skill_name)
            except KeyError:
                logger.warning("%s: skill '%s' not found", self.name, skill_name)
                continue
            missing = skill.validate_tools(self.PERMITTED_TOOLS)
            if missing:
                raise ValueError(
                    f"{self.name}: skill '{skill_name}' requires tools {missing} "
                    f"not in PERMITTED_TOOLS={self.PERMITTED_TOOLS}"
                )

    # ------------------------------------------------------------------
    # Conversation access (ADR-0003)
    # ------------------------------------------------------------------

    def _resolve_conversation(self, task_spec: dict[str, Any]) -> Conversation:
        """Return the agent's Conversation, resolving it through the runtime.

        When a runtime conversation store is wired (T4), the store owns the
        Conversation keyed by session/agent so it survives agent churn and
        API turns. Otherwise a per-instance fallback Conversation is used
        (standalone/test contexts).
        """
        if self._conversation is not None:
            return self._conversation

        store = None
        if self._runtime is not None:
            store = getattr(self._runtime, "conversation_store", None)
        if store is not None:
            session_id = task_spec.get("session_id", f"{self.name}")
            conv = store.get(session_id, self.name)
            self._conversation = conv
            return conv

        conv = Conversation(
            session_id=task_spec.get("session_id", f"{self.name}"),
            agent=self.name,
        )
        self._conversation = conv
        return conv

    def _session(self) -> Any | None:
        """The runtime's durable session object, or None (standalone/test)."""
        if self._runtime is not None:
            return getattr(self._runtime, "session", None)
        return None

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        config_path = Path(__file__).resolve().parent.parent / "config" / "dargus_config.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    # ------------------------------------------------------------------
    # Public entry point — unified PRA loop (ADR-0002)
    # ------------------------------------------------------------------

    def run(self, task_spec: dict[str, Any]) -> AgentReport:
        """Execute Perceive → Reason → Act loop until convergence.

        Every agent (Iris, D4Expert, Domain Expert) uses this single loop.
        Mode-gating controls which tools, skills, hooks, and system prompt
        are active for each round.
        """
        traces: list[CallTrace] = []
        round_num = 0
        converged = False
        data_gaps: list[str] = []
        bias_notes: list[str] = []
        findings: list = []

        # The Conversation is the single source of truth (ADR-0003): the loop
        # reads prior context from and appends every round to it. There is no
        # separate private history/act_cache buffer.
        conversation = self._resolve_conversation(task_spec)
        if not conversation.messages:
            # First round of a fresh conversation: record the task as the
            # opening user message. Subsequent turns append their own user
            # message via the same path (the Conversation persists on the
            # runtime), so cross-turn dialogue lives in one ordered log.
            conversation.add_user(
                json.dumps(task_spec, ensure_ascii=False),
                mode=self._mode,
            )

        while not converged and round_num < self.MAX_ROUNDS:
            # ── PERCEIVE: context assembly (no LLM call) ────────────
            perceive_t0 = time.monotonic()
            perceive_cache = self._perceive(task_spec, conversation, round_num)
            perceive_elapsed = int((time.monotonic() - perceive_t0) * 1000)

            perceive_trace = CallTrace(
                round=round_num,
                phase="perceive",
                output_summary=f"mode={self._mode}, tools={perceive_cache.get('tool_names', [])}",
                elapsed_ms=perceive_elapsed,
            )
            traces.append(perceive_trace)

            # Hook: PERCEIVE_END
            if self._hook_registry is not None:
                _ctx = HookContext(
                    runtime=self._runtime,
                    task_spec=task_spec,
                    session=self._session(),
                    agent=self,
                    round=round_num,
                    trace=perceive_trace,
                    extra={"perceive_cache": perceive_cache},
                )
                self._hook_registry.run(HookPoint.PERCEIVE_END, _ctx)

            # ── REASON: single LLM call ───────────────────────────
            reason_response, reason_trace = self._reason(
                task_spec, conversation, round_num, perceive_cache
            )
            traces.append(reason_trace)

            # Hook: REASON_END
            reason_ctx: HookContext | None = None
            if self._hook_registry is not None:
                reason_ctx = HookContext(
                    runtime=self._runtime,
                    task_spec=task_spec,
                    session=self._session(),
                    agent=self,
                    round=round_num,
                    trace=reason_trace,
                    extra={"reason_response": reason_response},
                )
                reason_ctx = self._hook_registry.run(HookPoint.REASON_END, reason_ctx)

            # ── ACT: execute or converge ──────────────────────────
            skip_act = reason_ctx.extra.get("skip_act", False) if reason_ctx else False
            if skip_act:
                # Mode-tag mismatch or hook veto — the round is skipped (the
                # warning was already injected into the next PERCEIVE round by
                # the mode-tag hook).
                act_output = {}
                act_traces: list[CallTrace] = []
            else:
                # Hook: ACT_START — pre-execution guard (workspace paths,
                # tool allowlist). Fired only when ACT will actually run.
                if self._hook_registry is not None:
                    _ctx = HookContext(
                        runtime=self._runtime,
                        task_spec=task_spec,
                        session=self._session(),
                        agent=self,
                        tools=self._tool_map(),
                        round=round_num,
                        trace=CallTrace(
                            round=round_num,
                            phase="act",
                            tool_called=reason_response.get("tool"),
                        ),
                        extra={"reason_response": reason_response},
                    )
                    self._hook_registry.run(HookPoint.ACT_START, _ctx)

                act_output, act_traces = self._act(reason_response, round_num, perceive_cache)

            traces.extend(act_traces)

            # Hook: ACT_END
            if self._hook_registry is not None and not skip_act:
                _ctx = HookContext(
                    runtime=self._runtime,
                    task_spec=task_spec,
                    session=self._session(),
                    agent=self,
                    tools=self._tool_map(),
                    round=round_num,
                    trace=act_traces[-1] if act_traces else None,
                    extra={"act_output": act_output},
                )
                self._hook_registry.run(HookPoint.ACT_END, _ctx)

            # ── Record the round in the Conversation ───────────────
            action = reason_response.get("action", "text")
            if action == "text":
                converged = True
                findings.append(reason_response.get("text", ""))
                conversation.add_assistant(reason_response.get("text", ""), mode=self._mode)
            elif action == "tool_call":
                tool_name = reason_response.get("tool", "")
                params = reason_response.get("params", {})
                # A failed Tool settles as an error Message in the log, not dropped.
                error = act_output.get("error") if isinstance(act_output, dict) else None
                conversation.add_tool(
                    ToolCall(name=tool_name, params=params),
                    ToolResult(output=act_output, error=error),
                    mode=self._mode,
                )

            # Track mode transitions
            if action == "tool_call" and reason_response.get("tool") == "switch_mode":
                new_mode = reason_response.get("params", {}).get("target", "")
                if new_mode:
                    logger.info("%s: mode transition %s → %s", self.name, self._mode, new_mode)

            # Hook: ROUND_END
            if self._hook_registry is not None:
                _ctx = HookContext(
                    runtime=self._runtime,
                    task_spec=task_spec,
                    session=self._session(),
                    agent=self,
                    round=round_num,
                    extra={},
                )
                self._hook_registry.run(HookPoint.ROUND_END, _ctx)

            round_num += 1

        confidence = self._compute_confidence(conversation)

        return AgentReport(
            agent_name=self.name,
            task_spec=task_spec,
            rounds=round_num,
            converged=converged,
            confidence=confidence,
            findings=findings,
            call_trace=traces,
            data_gaps=data_gaps,
            bias_notes=[n for n in bias_notes if n],
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _get_tool(self, name: str) -> Tool:
        return self._tool_registry.get(name)

    def _tool_map(self) -> dict[str, Tool]:
        """All registered tools keyed by name — for hook contexts."""
        try:
            return {tool.name: tool for tool in self._tool_registry.list_all()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the injected ReasoningLLM; stub when none is wired."""
        if self._reasoning_llm is not None:
            try:
                response = self._reasoning_llm.chat(
                    [
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_prompt),
                    ]
                )
                return response.content
            except Exception:
                logger.exception("%s: ReasoningLLM call failed", self.name)
                return json.dumps({"error": "llm_call_failed"})

        logger.warning("%s: no LLM configured, using stub", self.name)
        return self._llm_stub(system_prompt, user_prompt)

    def _llm_stub(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic stub when no LLM is available."""
        return json.dumps({"error": "no_llm_configured"})

    # ------------------------------------------------------------------
    # PERCEIVE — context assembly (no LLM call)
    # ------------------------------------------------------------------

    def _perceive(
        self,
        task_spec: dict,
        conversation: Conversation,
        round_num: int,
    ) -> dict[str, Any]:
        """Assemble the context blob for the next REASON round.

        Reads the current mode from ``self._mode``, looks up its ModeSpec,
        collects the mode's system prompt, tool definitions (as JSON Schema),
        and skill content. The dialogue context comes from the Conversation.

        No LLM call — pure data assembly.
        """
        # ── Mode spec lookup ───────────────────────────────────────
        mode_spec = self._mode_config.get(self._mode)
        system_prompt = ""
        mode_tool_names: list[str] = []
        mode_skill_names: list[str] = []

        if mode_spec is not None:
            system_prompt = mode_spec.system_prompt
            mode_tool_names = list(mode_spec.tools)
            mode_skill_names = list(mode_spec.skills)

        # ── Tool definitions (JSON Schema format) ──────────────────
        tool_defs: list[dict[str, Any]] = []
        for tool_name in mode_tool_names:
            try:
                tool = self._tool_registry.get(tool_name)
            except KeyError:
                continue
            # Skip tools that are mode-restricted and don't match current mode
            tool_modes: list[str] = getattr(tool, "_modes", [])
            if tool_modes and tool_modes != ["*"]:
                # Tool is restricted to specific modes
                pass  # mode_tool_names already filters

            params_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for p in tool.parameters:
                prop: dict[str, Any] = {"type": p.type, "description": p.description}
                if p.enum:
                    prop["enum"] = p.enum
                if p.default is not None:
                    prop["default"] = p.default
                params_schema["properties"][p.name] = prop
                if p.required:
                    params_schema["required"].append(p.name)

            tool_defs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params_schema,
                }
            )
            # Also include always-available tools (switch_mode)
            all_tools = self._tool_registry.list_all()
            for tool in all_tools:
                tool_modes: list[str] = getattr(tool, "_modes", [])
                if tool_modes == ["*"] and tool.name not in [t["name"] for t in tool_defs]:
                    params_schema_all: dict[str, Any] = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                    for p in tool.parameters:
                        prop_all: dict[str, Any] = {"type": p.type, "description": p.description}
                        if p.enum:
                            prop_all["enum"] = p.enum
                        if p.default is not None:
                            prop_all["default"] = p.default
                        params_schema_all["properties"][p.name] = prop_all
                        if p.required:
                            params_schema_all["required"].append(p.name)
                    tool_defs.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": params_schema_all,
                        }
                    )

        # ── Skill content ─────────────────────────────────────────
        skill_content: list[str] = []
        for skill_name in mode_skill_names:
            try:
                skill = self._skill_registry.get(skill_name)
                skill_content.append(skill.prompt)
            except KeyError:
                pass

        # ── Dialogue context: project the Conversation ────────────
        return {
            "system_prompt": system_prompt,
            "mode": self._mode,
            "mode_tool_names": mode_tool_names + self._always_available_tool_names(),
            "tool_definitions": tool_defs,
            "skill_content": skill_content,
            "task_spec": task_spec,
            "conversation": conversation,
            "llm_messages": conversation.to_llm_messages(),
            "round": round_num,
        }

    def _always_available_tool_names(self) -> list[str]:
        """Return tool names registered with ``_modes = ["*"]``."""
        names: list[str] = []
        for tool in self._tool_registry.list_all():
            if getattr(tool, "_modes", []) == ["*"]:
                names.append(tool.name)
        return names

    # ------------------------------------------------------------------
    # REASON — single LLM call
    # ------------------------------------------------------------------

    def _reason(
        self,
        task_spec: dict,
        conversation: Conversation,
        round_num: int,
        perceive_cache: dict[str, Any],
    ) -> tuple[dict, CallTrace]:
        """Forward the perceive cache to the LLM and parse the response.

        The LLM returns a mode-tagged JSON response:
          {"mode": "auto", "action": "text", "text": "Hello!"}
          {"mode": "auto", "action": "tool_call", "tool": "read_file",
           "params": {"path": "..."}}
        """
        t0 = time.monotonic()

        system_prompt = perceive_cache.get("system_prompt", self._build_reason_prompt())
        # The model-visible dialogue is the Conversation projected to LLM
        # messages — no JSON dump of a private history buffer (ADR-0003).
        llm_messages = perceive_cache.get("llm_messages", conversation.to_llm_messages())
        task_framing = json.dumps(
            {
                "task_spec": perceive_cache.get("task_spec", task_spec),
                "mode": perceive_cache.get("mode", self._mode),
                "available_tools": perceive_cache.get("tool_definitions", []),
                "available_skills": perceive_cache.get("skill_content", []),
                "round": round_num,
            },
            ensure_ascii=False,
        )
        dialogue = json.dumps(
            [{"role": m.role, "content": m.content} for m in llm_messages],
            ensure_ascii=False,
        )
        user_prompt = f"{task_framing}\n\nDialogue so far:\n{dialogue}"

        response = self._llm_call(system_prompt, user_prompt)
        try:
            reason_response = json.loads(response.strip())
        except json.JSONDecodeError:
            reason_response = {
                "mode": self._mode,
                "action": "text",
                "text": "I encountered an error processing your request.",
            }

        if "mode" not in reason_response:
            reason_response["mode"] = self._mode
        if "action" not in reason_response:
            reason_response["action"] = "text"
            reason_response["text"] = str(reason_response)

        elapsed = int((time.monotonic() - t0) * 1000)
        summary = (
            reason_response.get("text", "")[:200]
            if reason_response.get("action") == "text"
            else f"tool_call={reason_response.get('tool', '?')}"
        )
        trace = CallTrace(
            round=round_num,
            phase="reason",
            output_summary=summary,
            elapsed_ms=elapsed,
        )
        return reason_response, trace

    # ------------------------------------------------------------------
    # ACT — execute tool calls
    # ------------------------------------------------------------------

    def _act(
        self,
        reason_response: dict,
        round_num: int,
        perceive_cache: dict[str, Any] | None = None,
    ) -> tuple[dict, list[CallTrace]]:
        """Execute the tool call from the REASON response, if any.

        Mode-based tool enforcement: only tools listed in the perceive cache's
        ``mode_tool_names`` (plus always-available ``_modes=["*"]`` tools) may
        be called. Returns an error dict when the requested tool is not
        permitted in the current mode.

        Returns:
            (output_dict, traces_list). Output is empty for text responses.
        """
        if reason_response.get("action") != "tool_call":
            return {}, []

        tool_name = reason_response.get("tool", "")
        params = reason_response.get("params", {})
        traces: list[CallTrace] = []
        results: dict[str, Any] = {}

        # ── Mode-based tool authorization (ADR-0002) ───────────────
        allowed_tools: set[str] = set()
        if perceive_cache:
            allowed_tools.update(perceive_cache.get("mode_tool_names", []))
        if tool_name not in allowed_tools:
            # Always-available tool check (switch_mode is the canonical example)
            try:
                tool = self._get_tool(tool_name)
                tool_modes: list[str] = getattr(tool, "_modes", [])
                if tool_modes != ["*"]:
                    error_msg = f"Tool {tool_name!r} not permitted in mode {self._mode!r}"
                    results["output"] = {"error": error_msg}
                    traces.append(
                        CallTrace(
                            round=round_num,
                            phase="act",
                            tool_called=tool_name,
                            output_summary=error_msg,
                            elapsed_ms=0,
                            error=error_msg,
                        )
                    )
                    return results, traces
            except KeyError:
                error_msg = f"Tool {tool_name!r} not found"
                results["output"] = {"error": error_msg}
                traces.append(
                    CallTrace(
                        round=round_num,
                        phase="act",
                        tool_called=tool_name,
                        output_summary=error_msg,
                        elapsed_ms=0,
                        error=error_msg,
                    )
                )
                return results, traces

        t0 = time.monotonic()
        try:
            tool = self._get_tool(tool_name)
            step_result = tool.execute(**params)
            results["output"] = step_result
            error = None
        except KeyError:
            step_result = {"error": f"Tool {tool_name!r} not found"}
            results["output"] = step_result
            error = f"Tool {tool_name!r} not found"
        except Exception as exc:
            results["output"] = {"error": str(exc)}
            error = str(exc)

        # Surface failures at the top level too so the run loop can settle
        # an interrupted Tool as an error Message (T2 / #85).
        if error is not None:
            results["error"] = error

        elapsed = int((time.monotonic() - t0) * 1000)
        traces.append(
            CallTrace(
                round=round_num,
                phase="act",
                tool_called=tool_name,
                output_summary=str(results.get("output", ""))[:200],
                elapsed_ms=elapsed,
                error=error,
            )
        )

        return results, traces

    # ------------------------------------------------------------------
    # Prompt builders — override in subclasses (deprecated; use ModeSpec)
    # ------------------------------------------------------------------

    def _build_reason_prompt(self) -> str:
        """Default reason prompt — fallback when no ModeSpec system_prompt."""
        return (
            "You are a biomedical evidence analysis agent. "
            "Given a task specification and available tools, "
            "return a JSON response.\n\n"
            "Output format:\n"
            '{"mode": "<current_mode>", "action": "<text|tool_call>", '
            '"text": "<response if action is text>", '
            '"tool": "<tool name if action is tool_call>", '
            '"params": {}}}'
        )

    def _compute_confidence(self, conversation: Conversation) -> float:
        """Heuristic confidence from round count and convergence.

        Derived from the Conversation (ADR-0003) rather than a private
        history buffer. Each assistant Message contributes diminishing
        confidence.
        """
        n_rounds = sum(1 for m in conversation.messages if m.role == "assistant")
        if n_rounds == 0:
            return 0.0
        return min(0.9, n_rounds * 0.2)

    # ------------------------------------------------------------------
    # Backward-compat legacy methods (deprecated)
    # ------------------------------------------------------------------

    def register_skill(self, skill_path: str) -> None:
        """Register a SKILL extension (legacy compat)."""
        self.skills.add(Path(skill_path).stem)
        logger.info("%s registered skill %s", self.name, skill_path)

    def list_skills(self) -> list[str]:
        """List registered skills (legacy compat)."""
        return sorted(self.skills)

    def has_skill(self, skill_name: str) -> bool:
        """Check whether a skill is registered (legacy compat)."""
        return skill_name in self.skills

    def _project_dir(self, project_id: str) -> Path:
        root = Path(self.config.get("projects", {}).get("root_dir", "projects"))
        return root / project_id

    def _outputs_dir(self, project_id: str, layer: str, task_name: str) -> Path:
        out = self._project_dir(project_id) / "outputs" / layer / task_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_five_pack(
        self,
        project_id: str,
        layer: str,
        task_name: str,
        report: str,
        figures: dict[str, bytes] | None,
        data: dict[str, Any] | None,
        code: str | None,
        embedding: dict[str, Any],
    ) -> dict[str, str]:
        """Write the standard five-pack output (legacy compat)."""
        out = self._outputs_dir(project_id, layer, task_name)

        report_path = out / "report.md"
        report_path.write_text(report, encoding="utf-8")

        fig_paths: dict[str, str] = {}
        if figures:
            fig_dir = out / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in figures.items():
                path = fig_dir / fname
                path.write_bytes(content)
                fig_paths[fname] = str(path)

        data_paths: dict[str, str] = {}
        if data:
            data_dir = out / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            for fname, obj in data.items():
                path = data_dir / fname
                if isinstance(obj, bytes):
                    path.write_bytes(obj)
                else:
                    path.write_text(_to_csv_text(obj), encoding="utf-8")
                data_paths[fname] = str(path)

        code_path = out / "code" / "analysis.py"
        if code:
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")

        embedding_path = out / "level_embedding.json"
        embedding_path.write_text(json.dumps(embedding, indent=2), encoding="utf-8")

        return {
            "report": str(report_path),
            "figures": fig_paths,
            "data": data_paths,
            "code": str(code_path) if code else "",
            "level_embedding": str(embedding_path),
        }

    def _trace(self, project_id: str, task_id: str, event: str, details: dict[str, Any]) -> None:
        trace_dir = self._project_dir(project_id) / "logs" / "agent_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{self.name}.jsonl"
        line = json.dumps(
            {"timestamp": _now_iso(), "task_id": task_id, "event": event, "details": details},
            ensure_ascii=False,
        )
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _to_csv_text(obj: Any) -> str:
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj.to_csv(index=False)
    if isinstance(obj, list):
        return pd.DataFrame(obj).to_csv(index=False)
    raise TypeError(f"Cannot convert {type(obj)} to CSV text")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def new_task_id() -> str:
    return str(uuid.uuid4())
