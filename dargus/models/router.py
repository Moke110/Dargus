"""ModelRouter — phase-based backend selection with usage tracking."""

from __future__ import annotations

import copy

from dargus.models.reasoning import (
    LLMResponse,
    LLMUsage,
    Message,
    ReasoningBackend,
    ReasoningOptions,
)


class ModelRouter:
    """Routes reasoning calls to backends based on agent phase (planner/executor/critic).

    Tracks aggregate token usage and cost across all routed calls.
    """

    def __init__(self, backends: dict[str, ReasoningBackend]) -> None:
        """Initialise with a mapping of phase name to ReasoningBackend.

        Args:
            backends: Dict keyed by phase name (e.g. "planner", "executor", "critic").
        """
        self._backends = dict(backends)
        self._usage = LLMUsage()

    def route(
        self,
        phase: str,
        messages: list[Message],
        options: ReasoningOptions | None = None,
    ) -> LLMResponse:
        """Route a chat request to the backend for the given phase.

        Falls back to the "planner" backend if ``phase`` is not in the backends dict.

        Args:
            phase: The agent phase name ("planner", "executor", "critic", ...).
            messages: The chat messages to send.
            options: Optional reasoning parameter overrides.

        Returns:
            The LLMResponse from the selected backend.

        Raises:
            KeyError: If the phase is not found AND there is no "planner" fallback.
        """
        backend = self._backends.get(phase)
        if backend is None:
            backend = self._backends.get("planner")
        if backend is None:
            raise KeyError(f"No backend for phase '{phase}' and no 'planner' fallback available")

        response = backend.chat(messages, options)

        self._usage.input_tokens += response.usage.input_tokens
        self._usage.output_tokens += response.usage.output_tokens
        self._usage.cost += response.usage.cost

        return response

    @property
    def total_usage(self) -> LLMUsage:
        """Return a copy of the aggregate usage across all calls."""
        return copy.deepcopy(self._usage)

    def reset_usage(self) -> None:
        """Reset all tracked usage counters to zero."""
        self._usage = LLMUsage()
