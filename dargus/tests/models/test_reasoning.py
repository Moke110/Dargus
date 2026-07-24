"""Tests for ReasoningLLM, ReasoningBackend, and related dataclasses."""

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
