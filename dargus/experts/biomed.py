"""BiomedExpert — preclinical biology evidence assessment (skeleton)."""

from __future__ import annotations

from dargus.experts.base import Expert


class BiomedExpert(Expert):
    """Assesses preclinical wet-lab evidence: cell assays, organoids,
    organ-on-chip, ex-vivo tissue, and animal studies."""

    name = "BiomedExpert"
    system_prompt = (
        "You are BiomedExpert, a biomedical domain expert specializing in "
        "preclinical wet-lab evidence. Given a task specification and "
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

    SUPPORTED_LEVELS = (
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
    )
