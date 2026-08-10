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
from dargus.models.reasoning import LiteLLMBackend, Message, ReasoningLLM
from dargus.models.session import Session, SessionMetadata
from dargus.tools.base import Tool
from dargus.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Harness-equipped agent base class.

    Subclasses declare:
      - name: str
      - system_prompt: str
      - PERMITTED_TOOLS: list[str]
      - SUPPORTED_SKILLS: list[str]
    """

    name: str = "BaseAgent"
    system_prompt: str = ""

    # --- subclass overrides ---
    PERMITTED_TOOLS: list[str] = []
    SUPPORTED_SKILLS: list[str] = []
    MAX_ROUNDS: int = 5

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        runtime: Any | None = None,
    ):
        # Backward compat: if first positional arg is a dict, treat as config not name
        if isinstance(name, dict):
            config = name if config is None else config
            name = None

        if name is not None:
            self.name = name
        self.config = config or self._load_default_config()

        # DI or defaults — each injected value takes priority; None means "create default"
        self._tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._skill_registry = skill_registry if skill_registry is not None else SkillRegistry()

        # Reasoning LLM comes only from DI (the runtime's single model);
        # without it the agent runs in deterministic stub mode.
        self._reasoning_llm: ReasoningLLM | None = reasoning_llm

        # Back-reference to the owning DargusRuntime, injected by AgentFactory.
        # Runtime-dependent behaviour reads state off this; None in
        # standalone/test contexts.
        self._runtime: Any | None = runtime

        # The agent's Session (ADR-0005) — Iris instance state. The agent
        # reads and appends its own Session directly; the runtime holds no
        # per-session state.
        self._session = Session(SessionMetadata(agent=self.name))

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
    # Session access (ADR-0005)
    # ------------------------------------------------------------------

    def _resolve_session(self, task_spec: dict[str, Any]) -> Session:
        """Return the agent's own Session (ADR-0005).

        A live Session is Iris instance state: the agent reads and appends
        its own Session directly instead of resolving one through the
        runtime. The workspace root is seeded from the runtime's guard.
        """
        if self._runtime is not None and not self._session.metadata.workspace_root:
            self._session.metadata.workspace_root = self._runtime.workspace_root
        return self._session

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

        Every agent uses this single loop. The Session is Iris instance
        state (ADR-0005): the loop reads prior context from and appends
        every round to it.
        """
        traces: list[CallTrace] = []
        round_num = 0
        converged = False
        data_gaps: list[str] = []
        bias_notes: list[str] = []
        findings: list = []

        session = self._resolve_session(task_spec)
        # Each run() is one user Turn (ADR-0005): record the task as the
        # opening user prompt. Iris recalls prior Turns through the Session.
        session.add_user(json.dumps(task_spec, ensure_ascii=False))

        while not converged and round_num < self.MAX_ROUNDS:
            # ── PERCEIVE: context assembly (no LLM call) ────────────
            perceive_t0 = time.monotonic()
            perceive_cache = self._perceive(task_spec, session, round_num)
            perceive_elapsed = int((time.monotonic() - perceive_t0) * 1000)

            traces.append(
                CallTrace(
                    round=round_num,
                    phase="perceive",
                    output_summary=f"tools={perceive_cache.get('tool_names', [])}",
                    elapsed_ms=perceive_elapsed,
                )
            )

            # ── REASON: single LLM call ───────────────────────────
            reason_response, reason_trace = self._reason(
                task_spec, session, round_num, perceive_cache
            )
            traces.append(reason_trace)

            # ── ACT: execute or converge ──────────────────────────
            act_output, act_traces = self._act(reason_response, round_num, perceive_cache)
            traces.extend(act_traces)

            # ── Record the round in the Session ───────────────────
            action = reason_response.get("action", "text")
            if action == "text":
                converged = True
                findings.append(reason_response.get("text", ""))
                session.add_assistant(reason_response.get("text", ""))
            elif action == "tool_call":
                tool_name = reason_response.get("tool", "")
                params = reason_response.get("params", {})
                # A failed Tool settles as an error Round in the log, not dropped.
                error = act_output.get("error") if isinstance(act_output, dict) else None
                session.add_tool(
                    tool_name,
                    params=params,
                    output=act_output,
                    error=error,
                )

            round_num += 1

        # Close the current Turn: whether the run converged or exhausted
        # MAX_ROUNDS, its Rounds are no longer in flight and project coarse
        # from here on (ADR-0005 structural rule).
        session.close_current_turn()

        confidence = self._compute_confidence(session)

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
        """All registered tools keyed by name."""
        try:
            return {tool.name: tool for tool in self._tool_registry.list_all()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the injected ReasoningLLM; stub when none is wired.

        Falls back to the deterministic stub when no API key is configured
        (or the LLM call fails) so model-driven paths do not hang on a doomed
        network call in tests / key-less environments.
        """
        if self._reasoning_llm is not None and self._llm_available():
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

    def _llm_call_messages(self, system_prompt: str, messages: list[Message]) -> str:
        """Send a system prompt + projected messages to the ReasoningLLM.

        *messages* is the Session projected via ``projection()`` —
        the same role/content list the model consumes, so ``chat()`` receives
        the dialogue directly instead of a re-serialized JSON blob (SPEC-A).
        Falls back to the deterministic stub when no API key is configured.
        """
        if self._reasoning_llm is not None and self._llm_available():
            try:
                response = self._reasoning_llm.chat(
                    [Message(role="system", content=system_prompt)] + list(messages)
                )
                return response.content
            except Exception:
                logger.exception("%s: ReasoningLLM call failed", self.name)
                return json.dumps({"error": "llm_call_failed"})
        logger.warning("%s: no LLM configured, using stub", self.name)
        return self._llm_stub(system_prompt, "")

    def _llm_available(self) -> bool:
        """True when the reasoning LLM is usable (an API key is configured).

        The runtime may hold a ReasoningLLM even without a key (bootstrap
        keeps it for diagnostics); a key-less LiteLLM backend cannot complete
        a call, so the agent should fall back to its deterministic stub rather
        than hang on a doomed network call. Fake backends (tests) always count
        as available.
        """
        llm = self._reasoning_llm
        if llm is None:
            return False
        backend = getattr(llm, "_backend", None)
        if isinstance(backend, LiteLLMBackend):
            return bool(backend._api_key)
        return True

    def _llm_stub(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic stub when no LLM is available."""
        return json.dumps({"error": "no_llm_configured"})

    # ------------------------------------------------------------------
    # PERCEIVE — context assembly (no LLM call)
    # ------------------------------------------------------------------

    def _perceive(
        self,
        task_spec: dict,
        session: Session,
        round_num: int,
    ) -> dict[str, Any]:
        """Assemble the context blob for the next REASON round.

        Collects the agent's system prompt, the tool definitions for its
        ``PERMITTED_TOOLS`` (as JSON Schema), and skill content. The dialogue
        context comes from the Session.

        No LLM call — pure data assembly.
        """
        # ── Tool definitions (JSON Schema format) ──────────────────
        tool_defs: list[dict[str, Any]] = []
        for tool_name in self.PERMITTED_TOOLS:
            try:
                tool = self._tool_registry.get(tool_name)
            except KeyError:
                continue

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

        # ── Skill content ─────────────────────────────────────────
        skill_content: list[str] = []
        for skill_name in self.SUPPORTED_SKILLS:
            try:
                skill = self._skill_registry.get(skill_name)
                skill_content.append(skill.body)
            except KeyError:
                pass

        return {
            "system_prompt": self.system_prompt,
            "tool_names": [t["name"] for t in tool_defs],
            "tool_definitions": tool_defs,
            "skill_content": skill_content,
            "task_spec": task_spec,
            "conversation": session,
            "llm_messages": session.projection(),
            "round": round_num,
        }

    # ------------------------------------------------------------------
    # REASON — single LLM call
    # ------------------------------------------------------------------

    def _reason(
        self,
        task_spec: dict,
        session: Session,
        round_num: int,
        perceive_cache: dict[str, Any],
    ) -> tuple[dict, CallTrace]:
        """Forward the perceive cache to the LLM and parse the response.

        The LLM returns a JSON response:
          {"action": "text", "text": "Hello!"}
          {"action": "tool_call", "tool": "read_file", "params": {"path": "..."}}
        """
        t0 = time.monotonic()

        system_prompt = perceive_cache.get("system_prompt", self.system_prompt)
        # The model-visible dialogue is the Session projected to LLM
        # messages — the same role/content list ``chat()`` consumes (SPEC-A).
        llm_messages = perceive_cache.get("llm_messages", session.projection())
        # Task framing (task_spec, tools, skills, round) rides in the system
        # message; the projected dialogue is passed through verbatim.
        task_framing = json.dumps(
            {
                "task_spec": perceive_cache.get("task_spec", task_spec),
                "available_tools": perceive_cache.get("tool_definitions", []),
                "available_skills": perceive_cache.get("skill_content", []),
                "round": round_num,
            },
            ensure_ascii=False,
        )
        system_with_framing = f"{system_prompt}\n\n# Task framing\n{task_framing}"

        response = self._llm_call_messages(system_with_framing, llm_messages)
        try:
            reason_response = json.loads(response.strip())
        except json.JSONDecodeError:
            reason_response = {
                "action": "text",
                "text": "I encountered an error processing your request.",
            }

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

        Tool authorization: only tools in the agent's ``PERMITTED_TOOLS`` may
        be called. Returns an error dict when the requested tool is not
        permitted.

        Returns:
            (output_dict, traces_list). Output is empty for text responses.
        """
        if reason_response.get("action") != "tool_call":
            return {}, []

        tool_name = reason_response.get("tool", "")
        params = reason_response.get("params", {})
        traces: list[CallTrace] = []
        results: dict[str, Any] = {}

        # ── Tool authorization (PERMITTED_TOOLS) ──────────────────
        allowed_tools: set[str] = set(self.PERMITTED_TOOLS)
        if tool_name not in allowed_tools:
            error_msg = f"Tool {tool_name!r} not permitted for agent {self.name!r}"
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
        # an interrupted Tool as an error Message.
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
    # Confidence heuristic
    # ------------------------------------------------------------------

    def _compute_confidence(self, session: Session) -> float:
        """Heuristic confidence from round count and convergence.

        Derived from the Session (ADR-0005). Each assistant Round
        contributes diminishing confidence.
        """
        n_rounds = sum(1 for m in session.messages if m.role == "assistant")
        if n_rounds == 0:
            return 0.0
        return min(0.9, n_rounds * 0.2)

    # ------------------------------------------------------------------
    # Session end / persistence (ADR-0005)
    # ------------------------------------------------------------------

    def end(self) -> Path | None:
        """End this agent's Session and persist it to the archive.

        An ended Session is written once, never overwritten (the archive is
        append-only/immutable). Calling :meth:`end` repeatedly for the same
        Session is a no-op for the already-persisted archive entry.

        Returns:
            The written archive path, or ``None`` if there was nothing to
            write (no workspace root) or the archive refused the write.
        """
        session = self._session
        if session is None:
            return None

        root = session.metadata.workspace_root or self._session_workspace_root()
        if root:
            session.metadata.workspace_root = root
        if not root:
            logger.warning("%s: no workspace root — cannot persist session", self.name)
            return None

        from dargus.sessions.store import SessionStore

        session.close()
        store = SessionStore(workspace_root=root)
        return store.write(session)

    def close(self) -> None:
        """Explicit close hook (called by the runtime's AgentFactory and the
        REPL exit path). Persists the live Session — the archive is
        append-only, so a second close is a no-op."""
        try:
            self.end()
        except Exception:
            logger.exception("%s: failed to persist session on close", self.name)

    def _session_workspace_root(self) -> str:
        """Resolve the workspace root off the runtime (ADR-0005)."""
        if self._runtime is None:
            return ""
        return self._runtime.workspace_root


def new_task_id() -> str:
    return str(uuid.uuid4())
