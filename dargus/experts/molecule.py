"""MoleculeExpert — molecular-level evidence assessment (skeleton)."""

from __future__ import annotations

from dargus.experts.base import Expert


class MoleculeExpert(Expert):
    """Assesses drug physicochemical properties, drug-target relationships,
    medicinal chemistry, and formulation evidence at the molecular level."""

    name = "MoleculeExpert"
    system_prompt = (
        "You are MoleculeExpert, a biomedical domain expert specializing in "
        "evidence at the molecular level. Given a task specification and "
        "available tools, return a JSON response.\n\n"
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

    SUPPORTED_LEVELS = ("molecular", "molecular-sim")
