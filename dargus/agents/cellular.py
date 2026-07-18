"""CellAgent — cellular-level analysis (Phase 1)."""

from __future__ import annotations

from typing import Any

from dargus.agents.base import BaseAgent


class CellAgent(BaseAgent):
    """Placeholder for cellular-level analysis."""

    name = "CellAgent"

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        return {"status": "not_implemented", "agent": self.name}
