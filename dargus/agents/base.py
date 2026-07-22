"""BaseAgent — Harness skeleton with Planner -> Executor -> Critic execution loop."""

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
from dargus.knowledge.base import KnowledgeItem, KnowledgeRetriever
from dargus.tools.base import Tool
from dargus.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Harness-equipped agent base class.

    Subclasses declare:
      - PERMITTED_TOOLS: list[str]
      - PERMITTED_KNOWLEDGE: list[str]
      - SUPPORTED_SKILLS: list[str]
      - SUPPORTED_LEVELS: tuple[str, ...]
    """

    name: str = "BaseAgent"

    # --- subclass overrides ---
    PERMITTED_TOOLS: list[str] = []
    PERMITTED_KNOWLEDGE: list[str] = []
    SUPPORTED_SKILLS: list[str] = []
    SUPPORTED_LEVELS: tuple[str, ...] = ()
    MAX_ROUNDS: int = 5

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._load_default_config()
        self.skills: set[str] = set()
        self._tool_registry = ToolRegistry()
        self._skill_registry = SkillRegistry()
        self._knowledge_retrievers: dict[str, KnowledgeRetriever] = {}
        self._llm = self._init_llm()
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

    def _init_llm(self):
        """Create LLM client from config. Returns None if no LLM configured."""
        try:
            from dargus.llm import llm_from_config

            return llm_from_config(self.config)
        except Exception:
            return None

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        config_path = Path(__file__).resolve().parent.parent / "config" / "dargus_config.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    # ------------------------------------------------------------------
    # Public entry point — Harness loop
    # ------------------------------------------------------------------

    def run(self, task_spec: dict[str, Any]) -> AgentReport:
        """Execute the full Planner -> Executor -> Critic loop until convergence."""
        traces: list[CallTrace] = []
        history: list[dict] = []
        round_num = 0
        converged = False
        data_gaps: list[str] = []
        bias_notes: list[str] = []
        findings: list = []

        while not converged and round_num < self.MAX_ROUNDS:
            # --- Planner ---
            plan, plan_trace = self._plan(task_spec, history, round_num)
            traces.append(plan_trace)

            # --- Executor ---
            executor_output, exec_traces = self._execute(plan, round_num)
            traces.extend(exec_traces)

            # --- Critic ---
            verdict, critic_trace = self._criticize(plan, executor_output, history, round_num)
            traces.append(critic_trace)

            converged = verdict.get("converged", False)
            data_gaps.extend(verdict.get("gaps", []))
            bias_notes.append(verdict.get("next_round_guidance", ""))

            history.append(
                {
                    "round": round_num,
                    "plan": plan,
                    "executor_output": executor_output,
                    "verdict": verdict,
                }
            )

            if not converged and verdict.get("next_round_guidance"):
                task_spec = {**task_spec, "guidance": verdict["next_round_guidance"]}

            round_num += 1

        confidence = self._compute_confidence(history)

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
    # Knowledge injection
    # ------------------------------------------------------------------

    def _retrieve_knowledge(
        self,
        query: str,
        domain: str | None = None,
        biological_level: str | None = None,
    ) -> list[KnowledgeItem]:
        """Retrieve from all PERMITTED_KNOWLEDGE sources."""
        results: list[KnowledgeItem] = []
        for source_name in self.PERMITTED_KNOWLEDGE:
            retriever = self._knowledge_retrievers.get(source_name)
            if retriever is None:
                continue
            try:
                items = retriever.search(query, domain=domain, biological_level=biological_level)
                results.extend(items)
            except Exception:
                logger.exception("Knowledge retrieval failed for '%s'", source_name)
        return results

    def _format_knowledge_for_prompt(self, items: list[KnowledgeItem]) -> str:
        if not items:
            return ""
        lines = ["\n## Retrieved Knowledge\n"]
        for item in items:
            lines.append(
                f"- [{item.source}] {item.entity_type}/{item.entity_id}: {item.content[:500]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _get_tool(self, name: str) -> Tool:
        return self._tool_registry.get(name)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the configured LLM. Falls back to deterministic stub."""
        if self._llm is None:
            logger.warning("%s: no LLM configured, using stub", self.name)
            return self._llm_stub(system_prompt, user_prompt)
        try:
            return self._llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception:
            logger.exception("%s: LLM call failed", self.name)
            return self._llm_stub(system_prompt, user_prompt)

    def _llm_stub(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic stub when no LLM is available."""
        return json.dumps({"error": "no_llm_configured"})

    # ------------------------------------------------------------------
    # Phase methods — override in subclasses
    # ------------------------------------------------------------------

    def _plan(self, task_spec: dict, history: list[dict], round_num: int) -> tuple[dict, CallTrace]:
        """Planner phase: LLM generates a structured execution plan."""
        t0 = time.monotonic()
        knowledge = self._retrieve_knowledge(
            query=json.dumps(task_spec),
            biological_level=task_spec.get("biological_level"),
        )
        system_prompt = self._build_planner_prompt()
        user_prompt = json.dumps(
            {
                "task_spec": task_spec,
                "history": history,
                "available_skills": self.SUPPORTED_SKILLS,
                "available_tools": self.PERMITTED_TOOLS,
                "knowledge": [{"id": k.entity_id, "content": k.content[:300]} for k in knowledge],
            },
            ensure_ascii=False,
        )

        response = self._llm_call(system_prompt, user_prompt)
        try:
            plan = json.loads(response.strip())
        except json.JSONDecodeError:
            plan = {"goal": "parse_error", "steps": [], "expected_output": {}}

        elapsed = int((time.monotonic() - t0) * 1000)
        trace = CallTrace(
            round=round_num,
            phase="planner",
            knowledge_retrieved=[k.entity_id for k in knowledge],
            output_summary=str(plan.get("goal", ""))[:200],
            elapsed_ms=elapsed,
        )
        return plan, trace

    def _execute(self, plan: dict, round_num: int) -> tuple[dict, list[CallTrace]]:
        """Executor phase: ReAct loop over plan steps, calling Tools."""
        traces: list[CallTrace] = []
        results: dict[str, Any] = {}

        for i, step in enumerate(plan.get("steps", [])):
            skill_name = step.get("skill")
            tool_name = step.get("tool")
            params = step.get("params", {})

            if skill_name:
                try:
                    self._skill_registry.get(skill_name)
                except KeyError:
                    traces.append(
                        CallTrace(
                            round=round_num,
                            phase="executor",
                            skill_used=skill_name,
                            error=f"Skill '{skill_name}' not found",
                            elapsed_ms=0,
                        )
                    )
                    continue

            t0 = time.monotonic()
            if tool_name and tool_name in self.PERMITTED_TOOLS:
                try:
                    tool = self._get_tool(tool_name)
                    step_result = tool.execute(**params)
                    results[f"step_{i}"] = step_result
                    error = None
                except Exception as exc:
                    results[f"step_{i}"] = {"error": str(exc)}
                    error = str(exc)
            else:
                error = f"Tool '{tool_name}' not permitted or not found"

            elapsed = int((time.monotonic() - t0) * 1000)
            traces.append(
                CallTrace(
                    round=round_num,
                    phase="executor",
                    skill_used=skill_name,
                    tool_called=tool_name,
                    output_summary=str(results.get(f"step_{i}", ""))[:200],
                    elapsed_ms=elapsed,
                    error=error,
                )
            )

        return results, traces

    def _criticize(
        self, plan: dict, executor_output: dict, history: list[dict], round_num: int
    ) -> tuple[dict, CallTrace]:
        """Critic phase: LLM reviews results and judges convergence."""
        t0 = time.monotonic()
        system_prompt = self._build_critic_prompt()
        user_prompt = json.dumps(
            {
                "plan": plan,
                "executor_output": {k: str(v)[:300] for k, v in executor_output.items()},
                "history": [
                    {"round": h["round"], "verdict": h.get("verdict", {})} for h in history
                ],
            },
            ensure_ascii=False,
        )

        response = self._llm_call(system_prompt, user_prompt)
        try:
            verdict = json.loads(response.strip())
        except json.JSONDecodeError:
            verdict = {"converged": True, "confidence": 0.0, "gaps": ["critic_parse_error"]}

        elapsed = int((time.monotonic() - t0) * 1000)
        trace = CallTrace(
            round=round_num,
            phase="critic",
            output_summary=(
                f"converged={verdict.get('converged')}," f" confidence={verdict.get('confidence')}"
            ),
            elapsed_ms=elapsed,
        )
        return verdict, trace

    # ------------------------------------------------------------------
    # Prompt builders — override in subclasses
    # ------------------------------------------------------------------

    def _build_planner_prompt(self) -> str:
        return (
            "You are a planning agent for biomedical evidence analysis. "
            "Given a task specification, available skills, and tools, "
            "produce a structured execution plan as JSON.\n\n"
            "Output format:\n"
            '{"goal": "<string>", "steps": [{"skill": "<optional skill name>", '
            '"tool": "<optional tool name>", "params": {}, "rationale": "<string>"}], '
            '"expected_output": {}}'
        )

    def _build_critic_prompt(self) -> str:
        return (
            "You are a scientific critic. Review the execution output against the plan. "
            "Determine if the plan's expected_output has been satisfied. "
            "Identify remaining data gaps."
            "Judge whether another round would change conclusions.\n\n"
            "Output format:\n"
            '{"converged": <bool>, "confidence": <0.0-1.0>, '
            '"gaps": ["<string>"], "next_round_guidance": "<string or null>"}'
        )

    def _compute_confidence(self, history: list[dict]) -> float:
        if not history:
            return 0.0
        confidences = []
        for h in history:
            v = h.get("verdict", {})
            if isinstance(v, dict) and "confidence" in v:
                confidences.append(float(v["confidence"]))
        return sum(confidences) / len(confidences) if confidences else 0.0

    # ------------------------------------------------------------------
    # Backward-compat legacy methods (kept for ReportSearcher, etc.)
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
