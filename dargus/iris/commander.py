"""Iris — the orchestrating Agent, stripped to conversation + file access.

Task-specific prediction/ingestion orchestration has been removed to leave a
clean, general-purpose commander for the redo. Iris keeps the dialogue loop
(``ask``) and D-Base status reporting (``status``) inherited from the
BaseAgent PRA harness.
"""

from __future__ import annotations

import logging
from typing import Any

from dargus.agents.base import BaseAgent
from dargus.dbase import DBase
from dargus.dbase.paths import default_dargus_home, working_dbase
from dargus.dbase.store import DBaseStore

logger = logging.getLogger(__name__)


class Iris(BaseAgent):
    """The general-purpose commander Agent — dialogue via the PRA harness."""

    name = "Iris"
    system_prompt = (
        "You are Iris, the Dargus orchestrating agent. You can converse with "
        "the user and read files from the workspace. Given a task specification "
        "and available tools, return a JSON response.\n\n"
        "Output format:\n"
        '{"action": "<text|tool_call>", '
        '"text": "<response if action is text>", '
        '"tool": "<tool name if action is tool_call>", '
        '"params": {}}'
    )
    PERMITTED_TOOLS = ["read_file"]
    SUPPORTED_SKILLS = []  # Iris orchestrates; doesn't execute skills directly

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        agent_factory: Any | None = None,
        **di_kwargs: Any,
    ):
        super().__init__(config=config, **di_kwargs)
        self._agent_factory = agent_factory

        # ------------------------------------------------------------------
        # Skill loading: prefer the injected registry (AgentFactory path);
        # fall back to a local registry over the packaged skills directory.
        # ------------------------------------------------------------------
        if self._skill_registry is None:
            try:
                from pathlib import Path

                from dargus.agents.skill_registry import SkillRegistry

                _skills_dir = Path(__file__).resolve().parent.parent / "agents" / "skills"
                self._skill_registry = SkillRegistry(_skills_dir)
                _loaded = {s.name for s in self._skill_registry.list_all()}
                if _loaded:
                    logger.info("Iris loaded Skills: %s", sorted(_loaded))
                else:
                    logger.debug("Iris: no Skill files found in %s", _skills_dir)
            except Exception:
                logger.debug("Iris: SkillRegistry init skipped (skills dir missing)", exc_info=True)
                self._skill_registry = None

    def _global_manager(self) -> DBaseStore:
        dbase = DBase.global_instance()
        return DBaseStore(dbase)

    def status(self) -> dict[str, Any]:
        """Report global D-Base status."""
        dbase = DBase.global_instance()
        records = dbase.read_shards()
        return {
            "dargus_home": str(default_dargus_home()),
            "working_dbase": working_dbase(),
            "dbase_dir": str(dbase.dbase_dir),
            "n_records": len(records),
        }
