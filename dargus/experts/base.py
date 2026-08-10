"""Expert abstract base class — domain agents inheriting Harness from BaseAgent.

The domain assessment/extraction logic has been removed to leave a clean
skeleton for the redo. Each Expert subclass now declares only identity and
scope:

  - name: str
  - system_prompt: str (via BaseAgent)
  - SUPPORTED_LEVELS: which biological levels it can assess
  - PERMITTED_TOOLS: tools this expert may call during execution
  - SUPPORTED_SKILLS: skills this expert may load during planning
"""

from __future__ import annotations

from typing import Any

from dargus.agents.base import BaseAgent
from dargus.agents.skill_registry import SkillRegistry
from dargus.models.reasoning import ReasoningLLM
from dargus.tools.registry import ToolRegistry


class Expert(BaseAgent):
    """Domain expert skeleton with biological level declarations."""

    SUPPORTED_LEVELS: tuple[str, ...] = ()

    def __init__(
        self,
        dbase: Any = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
    ):
        super().__init__(
            config=config,
            reasoning_llm=reasoning_llm,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
        )
        self.dbase = dbase
