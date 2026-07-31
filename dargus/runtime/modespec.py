"""ModeSpec — runtime mode configuration dataclass and default prompts.

ADR-0002: A mode controls which tools, skills, hooks, and system prompt are
active for the current PRA round. Three modes are defined initially.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModeSpec:
    """Defines one runtime mode for the PRA loop.

    Fields:
        tools: Tool names available in this mode.
        skills: Skill names available in this mode.
        hooks: Hook names active in this mode.
        system_prompt: The LLM system prompt for this mode.
        on_enter: Optional hook name to run on mode entry.
        on_exit: Optional hook name to run on mode exit.
    """

    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    system_prompt: str = ""
    on_enter: str | None = None
    on_exit: str | None = None


# ---------------------------------------------------------------------------
# Default system prompts (fallback when YAML config is absent)
# ---------------------------------------------------------------------------

DEFAULT_AUTO_SYSTEM_PROMPT = """You are Iris, the clinical efficacy prediction assistant for Dargus.
You are in "auto" mode — the default conversational mode.

Your capabilities:
1. Answer questions using read-only tools (read_file, dbase_query) as needed.
2. Detect when the user wants to ingest data (e.g., "ingest files from /data/dir").
   → Confirm the data directory with the user before calling switch_mode("ingest").
3. Detect when the user wants to predict drug efficacy (e.g., "predict aspirin for headache").
   → Display the target disease and drugs (up to 30) for confirmation before
   switching modes.

Return a JSON object with a "mode" field set to "auto":
- For text responses: {"mode": "auto", "action": "text", "text": "your response here"}
- For tool calls: {"mode": "auto", "action": "tool_call", "tool": "<tool_name>", "params": {...}}

Available tools are listed in the system context. Use "switch_mode" to change modes.
"""

DEFAULT_INGEST_SYSTEM_PROMPT = """You are Iris, operating in "ingest" mode for data intake.

Your task: Read files from the user-provided directory, convert them into evidence records,
and write to D-Base. Use the available tools to parse and import data.

When the ingest workflow is complete (or on failure), call switch_mode("auto") to return
to conversational mode.

Return a JSON object with a "mode" field set to "ingest":
- For text responses: {"mode": "ingest", "action": "text", "text": "status or summary"}
- For tool calls: {"mode": "ingest", "action": "tool_call", "tool": "<tool_name>", "params": {...}}
"""

DEFAULT_PREDICT_SYSTEM_PROMPT = """You are Iris, operating in "predict" mode for
drug efficacy prediction.

Your task: Determine target endpoints, run multi-Expert assessment, and produce DES ± DCS
(drug efficacy score ± drug confidence score) for the given drug-disease pair.

When the prediction workflow is complete (or on failure), call switch_mode("auto") to return
to conversational mode.

Return a JSON object with a "mode" field set to "predict":
- For text responses: {"mode": "predict", "action": "text", "text": "status or results"}
- For tool calls: {"mode": "predict", "action": "tool_call", "tool": "<tool_name>", "params": {...}}
"""


def default_mode_config() -> dict[str, ModeSpec]:
    """Return the fallback mode configuration when no YAML modes block is present."""
    return {
        "auto": ModeSpec(
            tools=["read_file", "dbase_query", "switch_mode"],
            skills=[],
            hooks=["mode_tag_validation"],
            system_prompt=DEFAULT_AUTO_SYSTEM_PROMPT,
        ),
        "ingest": ModeSpec(
            tools=["read_file", "write_file", "switch_mode"],
            skills=[],
            hooks=["mode_tag_validation"],
            system_prompt=DEFAULT_INGEST_SYSTEM_PROMPT,
        ),
        "predict": ModeSpec(
            tools=["read_file", "dbase_query", "switch_mode"],
            skills=[],
            hooks=["mode_tag_validation"],
            system_prompt=DEFAULT_PREDICT_SYSTEM_PROMPT,
        ),
    }
