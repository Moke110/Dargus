"""AnimalAgent — animal-level analysis (Phase 1)."""

from __future__ import annotations

from typing import Any

from dargus.agents.base import BaseAgent


class AnimalAgent(BaseAgent):
    """Placeholder for animal model analysis."""

    name = "AnimalAgent"

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        return {"status": "not_implemented", "agent": self.name}
