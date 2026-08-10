"""D4Expert — Disease & Drug Development Director (skeleton)."""

from __future__ import annotations

from typing import Any

from dargus.agents.skill_registry import SkillRegistry
from dargus.experts.base import Expert
from dargus.models.reasoning import ReasoningLLM
from dargus.tools.registry import ToolRegistry


class D4Expert(Expert):
    """Disease & Drug Development Director.

    Holds broad-but-shallow knowledge across the full drug development
    stack. Does NOT perform technical orchestration (that's Iris's job).
    The coordination/synthesis methods have been removed to leave a clean
    skeleton for the redo.
    """

    name = "D4Expert"
    system_prompt = (
        "You are D4Expert, the Disease & Drug Development Director. "
        "You hold broad-but-shallow knowledge across the full drug "
        "development stack. Given a task specification and available tools, "
        "return a JSON response.\n\n"
        "Output format:\n"
        '{"action": "<text|tool_call>", '
        '"text": "<response if action is text>", '
        '"tool": "<tool name if action is tool_call>", '
        '"params": {}}'
    )
    # Task-specific tools (dbase_query, pubmed_search) were removed with the
    # task-specific code; Experts are silent skeletons until the redo.
    PERMITTED_TOOLS: list[str] = []
    SUPPORTED_SKILLS: list[str] = []

    SUPPORTED_LEVELS = (
        "molecular",
        "molecular-sim",
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
        "rct",
        "epi",
        "rct-sim",
    )

    def __init__(
        self,
        dbase: Any = None,
        config: dict[str, Any] | None = None,
        reasoning_llm: ReasoningLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        agent_factory: Any | None = None,
    ):
        super().__init__(
            dbase=dbase,
            config=config,
            reasoning_llm=reasoning_llm,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
        )
        self._agent_factory = agent_factory
