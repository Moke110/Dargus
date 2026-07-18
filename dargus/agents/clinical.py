"""ClinicalAgent — clinical-level analysis (Phase 1)."""

from __future__ import annotations

from typing import Any

from dargus.agents.base import BaseAgent


class ClinicAgent(BaseAgent):
    """Placeholder for clinical-level analysis."""

    name = "ClinicAgent"

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        return {"status": "not_implemented", "agent": self.name}
