"""ClinicExpert — rct and epidemiological evidence assessment (skeleton)."""

from __future__ import annotations

from dargus.experts.base import Expert


class ClinicExpert(Expert):
    """Assesses RCT, epidemiological, and post-market evidence.

    Covers rct, epi, and rct-sim levels with knowledge of trial design,
    medical statistics, and pharmacovigilance.
    """

    name = "ClinicExpert"
    system_prompt = (
        "You are ClinicExpert, a biomedical domain expert specializing in "
        "clinical trial and epidemiological evidence. Given a task "
        "specification and available tools, return a JSON response.\n\n"
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

    SUPPORTED_LEVELS = ("rct", "epi", "rct-sim")
