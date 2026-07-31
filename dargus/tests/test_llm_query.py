"""Tests for Iris.process_query() natural language parsing."""

import os

import pytest

from dargus.iris.commander import Iris, LLMCallError, NoLLMConfiguredError
from dargus.models.reasoning import LLMResponse, LLMUsage, Message


@pytest.fixture(autouse=True)
def _clear_api_key():
    """Prevent real .env API keys from leaking into tests."""
    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


class FakeReasoningBackend:
    """ReasoningBackend stub returning a controlled JSON response."""

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


def test_process_query_missing_llm_backend_raises():
    """Without an injected ReasoningLLM, raises NoLLMConfiguredError."""
    iris = Iris()
    with pytest.raises(NoLLMConfiguredError) as exc:
        iris.process_query("predict aspirin for headache")
    assert "No LLM backend configured" in str(exc.value)
    assert "dargus config set-api-key" in str(exc.value)


def test_process_query_predict_intent():
    """When LLM returns predict intent, Iris.predict is called."""
    import json

    response = json.dumps(
        {
            "intent": "predict",
            "drugs": ["aspirin"],
            "disease": "headache",
            "endpoints": [],
        }
    )

    result = _iris_with_llm(response).process_query("predict aspirin for headache")
    # Should route to predict — result is plain text (no Iris: prefix)
    assert "Iris:" not in result
    assert "No LLM backend configured" not in result


def test_process_query_status_intent():
    """When LLM returns status intent, Iris.status is called."""
    import json

    response = json.dumps({"intent": "status"})

    result = _iris_with_llm(response).process_query("what's the current status?")
    assert "D-Base status" in result
    assert "Records:" in result
    assert "Iris:" not in result


def test_process_query_clarify_intent():
    """When LLM returns clarify intent, the question is echoed."""
    import json

    response = json.dumps(
        {
            "intent": "clarify",
            "question": "Which disease are you interested in?",
        }
    )

    result = _iris_with_llm(response).process_query("predict aspirin")
    assert "Which disease are you interested in?" in result
    assert "Iris:" not in result


def test_process_query_chat_intent():
    """When LLM returns chat intent, the message is shown."""
    import json

    response = json.dumps(
        {
            "intent": "chat",
            "message": "I can help you predict drug efficacy. Try 'predict aspirin for headache'.",
        }
    )

    result = _iris_with_llm(response).process_query("hello")
    assert "I can help you predict" in result
    assert "Iris:" not in result


def test_process_query_invalid_json_raises():
    """Malformed LLM response raises LLMCallError (not a friendly fallback)."""
    with pytest.raises(LLMCallError):
        _iris_with_llm("not valid json {{{").process_query("blah")
