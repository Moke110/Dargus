"""Tests for Iris.process_query() natural language parsing."""

import os

import pytest

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


def test_process_query_missing_llm_backend_fallback():
    """Without an injected ReasoningLLM, returns config guidance."""
    iris = Iris()
    result = iris.process_query("predict aspirin for headache")
    assert "No LLM backend configured" in result
    assert "dargus config set-api-key" in result


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
    assert "Iris:" in result
    # Should route to predict — may succeed or fail depending on D-Base state
    # Either way it should not be the "no backend" fallback
    assert "No LLM backend configured" not in result


def test_process_query_status_intent():
    """When LLM returns status intent, Iris.status is called."""
    import json

    response = json.dumps({"intent": "status"})

    result = _iris_with_llm(response).process_query("what's the current status?")
    assert "D-Base status" in result
    assert "Records:" in result


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


def test_process_query_invalid_json_fallback():
    """Malformed LLM response returns a friendly error."""
    result = _iris_with_llm("not valid json {{{").process_query("blah")
    assert "I had trouble understanding" in result
