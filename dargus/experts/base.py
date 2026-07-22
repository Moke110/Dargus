"""Expert abstract base class — domain agents inheriting Harness from BaseAgent."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from dargus.agents.base import BaseAgent
from dargus.agents.report import AgentReport
from dargus.experts.protocol import ExpertContext, ExpertReport


class Expert(BaseAgent):
    """Domain expert with biological level declarations and delegation rules.

    Each Expert declares:
      - SUPPORTED_LEVELS: which biological levels it can assess
      - DELEGATION_RULES: {level: target_expert_name} for records outside scope
      - PERMITTED_TOOLS: tools this expert may call during execution
      - PERMITTED_KNOWLEDGE: knowledge sources this expert may query
      - SUPPORTED_SKILLS: skills this expert may load during planning
    """

    SUPPORTED_LEVELS: tuple[str, ...] = ()
    DELEGATION_RULES: dict[str, str] = {}

    def __init__(self, dbase: Any = None, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        self.dbase = dbase

    # ------------------------------------------------------------------
    # Core assess — called by run() for each round
    # ------------------------------------------------------------------

    @abstractmethod
    def assess(self, records: list[dict], context: ExpertContext) -> ExpertReport:
        """Assess evidence records and produce a structured report."""
        ...

    # ------------------------------------------------------------------
    # Record routing
    # ------------------------------------------------------------------

    def can_handle(self, record: dict) -> bool:
        level = self._read_biological_level(record)
        return level in self.SUPPORTED_LEVELS

    def delegate_target(self, record: dict) -> str | None:
        level = self._read_biological_level(record)
        if level is None:
            return None
        return self.DELEGATION_RULES.get(level)

    def _read_biological_level(self, record: dict) -> str | None:
        return record.get("biological_level")

    def _read_field(self, record: dict, field_name: str) -> Any:
        return record.get(field_name)

    # ------------------------------------------------------------------
    # Prompt overrides — domain-specific system prompts
    # ------------------------------------------------------------------

    def _build_planner_prompt(self) -> str:
        return (
            f"You are {self.name}, a biomedical domain expert specializing in "
            f"evidence at biological levels: {', '.join(self.SUPPORTED_LEVELS)}.\n"
            "Given a task specification and available skills/tools/knowledge, "
            "produce a structured execution plan as JSON.\n\n"
            "Output format:\n"
            '{"goal": "<string>", "steps": [{"skill": "<optional skill name>", '
            '"tool": "<optional tool name>", "params": {}, "rationale": "<string>"}], '
            '"expected_output": {}}'
        )

    def _build_critic_prompt(self) -> str:
        return (
            f"You are {self.name}, critically reviewing execution results. "
            "Determine if findings are sufficient. Identify remaining gaps. "
            "Judge whether another analysis round would change conclusions.\n\n"
            "Output format:\n"
            '{"converged": <bool>, "confidence": <0.0-1.0>, '
            '"gaps": ["<string>"], "next_round_guidance": "<string or null>"}'
        )

    # ------------------------------------------------------------------
    # run() override — adapts Expert lifecycle into BaseAgent Harness
    # ------------------------------------------------------------------

    def run(self, task_spec: dict[str, Any]) -> AgentReport:
        """Run Expert assessment via Harness, then call assess() on gathered evidence."""
        report = super().run(task_spec)
        records = task_spec.get("_records", [])
        if records:
            ctx = ExpertContext(
                drug_ids=task_spec.get("drug_ids", []),
                disease_id=task_spec.get("disease_id", ""),
                endpoints=task_spec.get("endpoints", []),
                round=report.rounds,
            )
            expert_report = self.assess(records, ctx)
            report.findings = [expert_report]
        return report
