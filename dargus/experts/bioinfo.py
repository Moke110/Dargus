"""BioinfoExpert — high-throughput/omics evidence assessment (skeleton)."""

from __future__ import annotations

from dargus.experts.base import Expert


class BioinfoExpert(Expert):
    """Assesses high-throughput and omics data across all biological levels."""

    name = "BioinfoExpert"
    system_prompt = (
        "You are BioinfoExpert, a biomedical domain expert specializing in "
        "high-throughput and omics data. Given a task specification and "
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
