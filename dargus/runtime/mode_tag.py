"""Mode-tag validation hook — validates LLM response mode matches runtime mode.

ADR-0002: The REASON_END hook checks that the LLM response ``mode`` field
matches ``runtime.mode``. On mismatch, ACT is skipped and a warning is
injected into the next PERCEIVE context for corrective feedback.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModeTagValidationHook:
    """REASON_END hook: validate LLM response mode-tag.

    On match: returns context unchanged.
    On mismatch: sets ``context.extra["skip_act"] = True`` and
    ``context.extra["mode_tag_warning"]`` with a human-readable message.
    The warning is surfaced in the next PERCEIVE round.
    """

    def __call__(self, context: Any) -> Any:
        runtime = context.runtime
        if runtime is None:
            return context  # No runtime → nothing to validate

        reason_response = context.extra.get("reason_response", {})
        # Validate against the acting agent's own mode (a Subagent runs in its
        # own mode, not the runtime's), falling back to the runtime's mode.
        agent = context.agent
        expected_mode = getattr(agent, "_mode", None) or runtime.mode
        response_mode = reason_response.get("mode", "")

        if not response_mode:
            # No mode in response — treat as auto (don't block)
            logger.debug("ModeTagValidationHook: no mode field in LLM response — allowing")
            return context

        if response_mode == expected_mode:
            # Match — ACT may proceed
            return context

        # Mismatch — block ACT, inject warning
        warning = (
            f"Mode tag mismatch: LLM responded with mode={response_mode!r} "
            f"but the agent is in mode={expected_mode!r}. "
            "Please ensure your response uses the correct mode field."
        )
        context.extra["skip_act"] = True
        context.extra["mode_tag_mismatch"] = True
        context.extra["mode_tag_warning"] = warning
        logger.warning(
            "ModeTagValidationHook: mode mismatch — expected %r, got %r — blocking ACT",
            expected_mode,
            response_mode,
        )
        return context
