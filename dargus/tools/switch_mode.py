"""switch_mode Tool — always-available mode transition for PRA loop.

ADR-0002: The ``switch_mode`` tool is registered with ``_modes = ["*"]`` so it
is always available regardless of the current mode. It calls
``runtime.switch_mode(target)`` which fires on_enter/on_exit hooks.
"""

from __future__ import annotations

import logging
from typing import Any

from dargus.tools.base import Tool, ToolParam

logger = logging.getLogger(__name__)


def _impl_switch_mode(runtime: Any, target: str) -> dict:
    """Implementation for ``switch_mode``."""
    ok = runtime.switch_mode(target)
    if ok:
        return {"switched_to": target, "current_mode": runtime.mode}
    return {"error": f"Unknown mode {target!r}", "current_mode": runtime.mode}


def make_switch_mode_tool(runtime: Any) -> Tool:
    """Create a fully wired ``switch_mode`` Tool.

    The tool is always available regardless of mode (``_modes = ["*"]``).

    Args:
        runtime: The DargusRuntime instance.
    """
    tool = Tool(
        name="switch_mode",
        description=(
            "Switch the runtime to a different mode. "
            "Available modes: auto (conversational), "
            "ingest (data intake), predict (efficacy prediction)."
        ),
        parameters=[
            ToolParam(
                name="target",
                type="string",
                required=True,
                description="The target mode name: auto, ingest, or predict.",
                enum=["auto", "ingest", "predict"],
            ),
        ],
        output={
            "type": "object",
            "properties": {
                "switched_to": {"type": "string"},
                "current_mode": {"type": "string"},
            },
        },
        timeout_ms=5_000,
        fallback="error",
        side_effect="read",
    )
    # Mark as always-available regardless of mode
    tool._modes = ["*"]
    # Bind closure capturing runtime
    tool.bind(lambda **kwargs: _impl_switch_mode(runtime, **kwargs))
    return tool
