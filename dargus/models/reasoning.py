"""ReasoningLLM — the unified LLM abstraction for agent reasoning phases."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single chat message with role and content."""

    role: str
    content: str


@dataclass
class ReasoningOptions:
    """Optional overrides for a reasoning call."""

    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None


@dataclass
class LLMUsage:
    """Token usage and cost tracking for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class LLMResponse:
    """The result of a reasoning call."""

    content: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ReasoningBackend(Protocol):
    """Protocol for reasoning backends — the actual model invocation layer."""

    def chat(self, messages: list[Message], options: ReasoningOptions | None = None) -> LLMResponse:
        """Send messages to the model and return a structured response."""
        ...


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------


class LiteLLMBackend:
    """ReasoningBackend that delegates to litellm for multi-provider support."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def chat(self, messages: list[Message], options: ReasoningOptions | None = None) -> LLMResponse:
        """Call litellm with the given messages and return an LLMResponse.

        Raises RuntimeError on failure with context about the provider/model.
        """
        opts = options or ReasoningOptions()

        litellm_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            import litellm

            model_id = f"{self._provider}/{self._model}"

            kwargs: dict = {
                "model": model_id,
                "messages": litellm_messages,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["api_base"] = self._base_url
            if opts.temperature is not None:
                kwargs["temperature"] = opts.temperature
            if opts.max_tokens is not None:
                kwargs["max_tokens"] = opts.max_tokens
            if opts.stop_sequences:
                kwargs["stop"] = opts.stop_sequences

            response = litellm.completion(**kwargs)

            content = response.choices[0].message.content or ""
            usage_data = getattr(response, "usage", None)
            input_tokens = getattr(usage_data, "prompt_tokens", 0) if usage_data else 0
            output_tokens = getattr(usage_data, "completion_tokens", 0) if usage_data else 0
            finish_reason = getattr(response.choices[0], "finish_reason", "stop") or "stop"

            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass

            return LLMResponse(
                content=content,
                usage=LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                ),
                model=f"{self._provider}/{self._model}",
                finish_reason=finish_reason,
            )

        except Exception as exc:
            raise RuntimeError(
                f"LiteLLMBackend call failed for {self._provider}/{self._model}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# ReasoningLLM facade
# ---------------------------------------------------------------------------


class ReasoningLLM:
    """Unified reasoning interface used by agents in P-R-A cycles.

    Delegates to a ReasoningBackend and supports session-scoped variants.
    """

    def __init__(
        self,
        backend: ReasoningBackend,
        default_options: ReasoningOptions | None = None,
    ) -> None:
        self._backend = backend
        self._default_options = default_options or ReasoningOptions()

    def chat(self, messages: list[Message], options: ReasoningOptions | None = None) -> LLMResponse:
        """Send messages to the reasoning backend, merging default and per-call options.

        Per-call options override defaults on a per-field basis.
        """
        merged = ReasoningOptions(
            temperature=(
                options.temperature
                if (options and options.temperature is not None)
                else self._default_options.temperature
            ),
            max_tokens=(
                options.max_tokens
                if (options and options.max_tokens is not None)
                else self._default_options.max_tokens
            ),
            stop_sequences=(
                options.stop_sequences
                if (options and options.stop_sequences is not None)
                else self._default_options.stop_sequences
            ),
        )
        return self._backend.chat(messages, merged)

    def with_session(self, session_id: str) -> "ReasoningLLM":
        """Return a session-scoped ReasoningLLM.

        Currently a stub — stores session_id and returns self. Session-scoped
        behaviour (conversation history, caching) is a future concern.
        """
        self._session_id = session_id
        return self
