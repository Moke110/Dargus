"""Runtime-level error classes.

These were previously in dargus.iris.commander, but the PRA loop and mode
system make them runtime-level concerns rather than Iris-specific concerns.
"""

from __future__ import annotations


class NoLLMConfiguredError(Exception):
    """Raised when an agent receives a query but no LLM backend is wired."""

    def __init__(self) -> None:
        super().__init__(
            "No LLM backend configured.\n\n"
            "Set your API key with:\n"
            "  dargus config set-api-key <provider> <key>\n\n"
            "Or use CLI subcommands directly:\n"
            "  dargus predict --drugs aspirin --disease headache\n"
            "  dargus status\n"
            "  dargus --help"
        )


class LLMCallError(Exception):
    """Raised when the LLM call fails (network, API, or malformed response)."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"LLM call failed: {detail}\n\n"
            "Check your API key and network. Use /config to reconfigure."
        )
