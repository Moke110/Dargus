"""Tests for ReasoningLLM, ReasoningBackend, and related dataclasses."""

from unittest.mock import MagicMock, patch

import pytest

from dargus.models.reasoning import (
    LiteLLMBackend,
    LLMResponse,
    LLMUsage,
    Message,
    ReasoningLLM,
    ReasoningOptions,
)


class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_equality(self):
        m1 = Message(role="user", content="hi")
        m2 = Message(role="user", content="hi")
        assert m1 == m2


class TestLLMUsage:
    """Tests for LLMUsage dataclass."""

    def test_default_values(self):
        usage = LLMUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost == 0.0

    def test_custom_values(self):
        usage = LLMUsage(input_tokens=100, output_tokens=50, cost=0.003)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cost == 0.003


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_default_values(self):
        resp = LLMResponse(content="test")
        assert resp.content == "test"
        assert resp.usage == LLMUsage()
        assert resp.model == ""
        assert resp.finish_reason == "stop"

    def test_full_response(self):
        usage = LLMUsage(input_tokens=10, output_tokens=5)
        resp = LLMResponse(
            content="result", usage=usage, model="anthropic/claude", finish_reason="stop"
        )
        assert resp.finish_reason == "stop"
        assert resp.model == "anthropic/claude"


class TestReasoningOptions:
    """Tests for ReasoningOptions dataclass."""

    def test_defaults_are_none(self):
        opts = ReasoningOptions()
        assert opts.temperature is None
        assert opts.max_tokens is None
        assert opts.stop_sequences is None

    def test_custom_values(self):
        opts = ReasoningOptions(temperature=0.5, max_tokens=2000, stop_sequences=["END"])
        assert opts.temperature == 0.5
        assert opts.max_tokens == 2000
        assert opts.stop_sequences == ["END"]


class MockReasoningBackend:
    """Minimal mock ReasoningBackend for testing."""

    def __init__(self, response_content: str = "mock response"):
        self._response_content = response_content
        self.last_messages: list[Message] = []
        self.last_options: ReasoningOptions | None = None
        self.call_count = 0

    def chat(self, messages: list[Message], options: ReasoningOptions | None = None) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_options = options
        return LLMResponse(
            content=self._response_content,
            usage=LLMUsage(input_tokens=10, output_tokens=5, cost=0.001),
            model="mock/model",
            finish_reason="stop",
        )


class TestReasoningLLM:
    """Tests for ReasoningLLM facade."""

    def test_chat_delegates_to_backend(self):
        backend = MockReasoningBackend()
        llm = ReasoningLLM(backend)
        messages = [Message(role="user", content="hello")]
        resp = llm.chat(messages)

        assert backend.call_count == 1
        assert backend.last_messages == messages
        assert resp.content == "mock response"
        assert resp.model == "mock/model"

    def test_chat_merges_options_correctly(self):
        """Per-call options override defaults, missing per-call fields use defaults."""
        backend = MockReasoningBackend()
        default_opts = ReasoningOptions(temperature=0.0, max_tokens=4096)
        llm = ReasoningLLM(backend, default_options=default_opts)

        llm.chat(
            [Message(role="user", content="hi")],
            options=ReasoningOptions(temperature=0.8),
        )

        assert backend.last_options is not None
        assert backend.last_options.temperature == 0.8  # overridden
        assert backend.last_options.max_tokens == 4096  # from default

    def test_chat_no_default_options(self):
        backend = MockReasoningBackend()
        llm = ReasoningLLM(backend)
        resp = llm.chat([Message(role="user", content="x")])
        assert resp.content == "mock response"

    def test_with_session_returns_self(self):
        backend = MockReasoningBackend()
        llm = ReasoningLLM(backend)
        result = llm.with_session("session-123")
        assert result is llm

    def test_with_session_stores_session_id(self):
        backend = MockReasoningBackend()
        llm = ReasoningLLM(backend)
        llm.with_session("session-abc")
        assert getattr(llm, "_session_id", None) == "session-abc"


class TestLiteLLMBackend:
    """Tests for LiteLLMBackend (basic constructor + error handling)."""

    def test_constructor_stores_params(self):
        backend = LiteLLMBackend(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-test",
            base_url="https://api.example.com",
        )
        assert backend._provider == "anthropic"
        assert backend._model == "claude-sonnet-4"
        assert backend._api_key == "sk-test"
        assert backend._base_url == "https://api.example.com"

    def test_constructor_minimal_params(self):
        backend = LiteLLMBackend(provider="openai", model="gpt-4o", api_key="sk-123")
        assert backend._base_url is None

    def test_chat_returns_llm_response_with_usage(self):
        """chat() parses litellm response and returns LLMResponse with usage info."""
        backend = LiteLLMBackend(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-test",
        )
        messages = [Message(role="user", content="hello")]

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 42
        mock_usage.completion_tokens = 17

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, world!"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("litellm.completion", return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.0015):
                resp = backend.chat(messages)

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello, world!"
        assert resp.model == "anthropic/claude-sonnet-4"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 42
        assert resp.usage.output_tokens == 17
        assert resp.usage.cost == 0.0015

    def test_chat_handles_missing_usage_gracefully(self):
        """chat() handles response with no usage field gracefully."""
        backend = LiteLLMBackend(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-test",
        )
        messages = [Message(role="user", content="hi")]

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        # Make usage evaluate as None (falsy) so the code path handles it
        mock_response.usage = None

        with patch("litellm.completion", return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.0):
                resp = backend.chat(messages)

        assert resp.content == "ok"
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0

    def test_chat_handles_none_content(self):
        """chat() handles response with None message content."""
        backend = LiteLLMBackend(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
        )
        messages = [Message(role="user", content="test")]

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.0):
                resp = backend.chat(messages)

        assert resp.content == ""
        assert resp.finish_reason == "length"

    def test_chat_passes_options_to_litellm(self):
        """chat() passes temperature, max_tokens, and stop_sequences to litellm."""
        backend = LiteLLMBackend(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-test",
        )
        messages = [Message(role="user", content="test")]

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            with patch("litellm.completion_cost", return_value=0.0):
                backend.chat(
                    messages,
                    options=ReasoningOptions(
                        temperature=0.7,
                        max_tokens=512,
                        stop_sequences=["END"],
                    ),
                )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["stop"] == ["END"]

    def test_chat_passes_base_url_when_set(self):
        """chat() passes api_base to litellm when base_url is set."""
        backend = LiteLLMBackend(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://custom.api.com",
        )
        messages = [Message(role="user", content="hi")]

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            with patch("litellm.completion_cost", return_value=0.0):
                backend.chat(messages)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["api_base"] == "https://custom.api.com"

    def test_chat_wraps_errors_as_runtime_error(self):
        """chat() wraps litellm exceptions in RuntimeError with provider/model context."""
        backend = LiteLLMBackend(
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-bad",
        )
        messages = [Message(role="user", content="hello")]

        with patch(
            "litellm.completion",
            side_effect=Exception("API error: 401 Unauthorized"),
        ):
            with pytest.raises(
                RuntimeError,
                match="LiteLLMBackend call failed for anthropic/claude-sonnet-4",
            ):
                backend.chat(messages)
