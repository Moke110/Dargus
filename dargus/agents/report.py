"""Agent reporting types — AgentReport and CallTrace dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CallTrace:
    round: int
    phase: str  # "perceive" | "reason" | "act"
    skill_used: str | None = None
    tool_called: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    elapsed_ms: int = 0
    error: str | None = None


@dataclass
class AgentReport:
    agent_name: str
    task_spec: dict
    rounds: int
    converged: bool
    confidence: float
    findings: list = field(default_factory=list)
    call_trace: list[CallTrace] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    bias_notes: list[str] = field(default_factory=list)
