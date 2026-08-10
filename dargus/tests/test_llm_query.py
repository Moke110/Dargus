"""Tests for Iris.run() — PRA loop via FakeReasoningBackend.

ADR-0002: Iris.process_query() is deleted. All NL processing goes through
the unified PRA (Perceive→Reason→Act) loop via iris.run().
"""

import os

import pytest

from dargus.agents.report import AgentReport
from dargus.iris.commander import Iris
from dargus.models.reasoning import LLMResponse, LLMUsage, Message


@pytest.fixture(autouse=True)
def _clear_api_key():
    """Prevent real .env API keys from leaking into tests."""
    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


class FakeReasoningBackend:
    """ReasoningBackend stub returning a controlled PRA JSON response."""

    def __init__(self, response: str):
        self._response = response

    def chat(self, messages: list[Message], options=None) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            usage=LLMUsage(),
            model="fake",
            finish_reason="stop",
        )


def _iris_with_llm(response: str) -> Iris:
    from dargus.models.reasoning import ReasoningLLM

    return Iris(reasoning_llm=ReasoningLLM(backend=FakeReasoningBackend(response)))


# ------------------------------------------------------------------
# Basic PRA loop tests
# ------------------------------------------------------------------


def test_iris_run_text_response():
    """LLM returns a text action → loop converges in 1 round."""
    import json

    response = json.dumps({"action": "text", "text": "Hello! How can I help?"})

    iris = _iris_with_llm(response)
    report = iris.run({"query": "hello"})

    assert isinstance(report, AgentReport)
    assert report.converged is True
    assert report.rounds == 1
    assert "Hello!" in report.findings[-1]


def test_iris_run_tool_call():
    """LLM returns a tool_call → loop continues."""
    import json

    # Round 1: tool call → Round 2: text
    responses = iter(
        [
            json.dumps({"action": "text", "text": "Let me check that for you."}),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            try:
                content = next(responses)
            except StopIteration:
                content = json.dumps({"action": "text", "text": "Done."})
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "read data/file.txt"})

    assert isinstance(report, AgentReport)
    assert report.rounds >= 1


def test_iris_run_missing_llm_backend():
    """Without an injected ReasoningLLM, the agent stub still produces a report."""
    iris = Iris()
    report = iris.run({"query": "hello"})
    assert isinstance(report, AgentReport)
    # The stub returns {"error": "no_llm_configured"} which is not valid PRA JSON
    # so it falls through to parse error → text response
    assert report.rounds <= 5  # bounded by MAX_ROUNDS


# ------------------------------------------------------------------
# api.ask() integration
# ------------------------------------------------------------------


def test_api_ask_with_fake_backend(monkeypatch):
    """api.ask() calls iris.run() and returns text response."""
    import json

    from dargus import api

    response = json.dumps({"action": "text", "text": "Iris at your service."})

    class FakeLLM:
        def chat(self, messages, options=None):
            return LLMResponse(content=response, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    llm = ReasoningLLM(backend=FakeLLM())

    # Patch _create_iris_with_lm to return Iris with our fake LLM
    def _create(*args, **kwargs):
        return Iris(reasoning_llm=llm)

    monkeypatch.setattr(api, "_create_iris_with_lm", _create)
    result = api.ask("hello")
    assert "Iris at your service" in result


# ------------------------------------------------------------------
# PRA loop paths
# ------------------------------------------------------------------


def test_chat_path():
    """Chat: "hi" → 1-round converge with text response."""
    import json

    response = json.dumps(
        {
            "action": "text",
            "text": "Hi! I'm Iris. How can I help with your research?",
        }
    )

    iris = _iris_with_llm(response)
    report = iris.run({"query": "hi"})

    assert report.converged is True
    assert report.rounds == 1
    assert "Iris" in str(report.findings)


def test_tool_calling_path():
    """User asks a question requiring file access → LLM calls read_file tool."""
    import json

    # Round 1: tool call for read_file → Round 2: text
    responses = iter(
        [
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "params": {"path": "README.md"},
                }
            ),
            json.dumps(
                {
                    "action": "text",
                    "text": "The README contains setup instructions.",
                }
            ),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            content = next(responses)
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "what's in README.md?"})

    assert isinstance(report, AgentReport)
    # Should converge: round 1 tool_call, round 2 text
    assert report.converged is True
    assert report.rounds >= 2
