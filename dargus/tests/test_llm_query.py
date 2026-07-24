"""Tests for Iris.process_query() natural language parsing."""

import os
from unittest.mock import patch

import pytest

from dargus.iris.commander import Iris


@pytest.fixture(autouse=True)
def _clear_api_key():
    """Prevent real .env API keys from leaking into tests."""
    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


class FakeLLMBackend:
    """Returns a controlled JSON response for testing."""

    def __init__(self, response: str):
        self._response = response

    def complete(self, prompt: str, **kwargs):
        return self._response


def _fake_backend(response: str):
    return lambda config: FakeLLMBackend(response)


def test_process_query_missing_llm_backend_fallback():
    """Without LLM backend (None from llm_from_config), returns config guidance."""
    iris = Iris()
    with patch("dargus.models.compat.llm_from_config", return_value=None):
        result = iris.process_query("predict aspirin for headache")
    assert "No LLM backend configured" in result
    assert "dargus config set-api-key" in result


def test_process_query_predict_intent():
    """When LLM returns predict intent, Iris.predict is called."""
    import json

    iris = Iris()
    response = json.dumps(
        {
            "intent": "predict",
            "drugs": ["aspirin"],
            "disease": "headache",
            "endpoints": [],
        }
    )

    with patch(
        "dargus.models.compat.llm_from_config",
        _fake_backend(response),
    ):
        result = iris.process_query("predict aspirin for headache")
        assert "Iris:" in result
        # Should route to predict — may succeed or fail depending on D-Base state
        # Either way it should not be the "no backend" fallback
        assert "No LLM backend configured" not in result


def test_process_query_status_intent():
    """When LLM returns status intent, Iris.status is called."""
    import json

    iris = Iris()
    response = json.dumps({"intent": "status"})

    with patch(
        "dargus.models.compat.llm_from_config",
        _fake_backend(response),
    ):
        result = iris.process_query("what's the current status?")
        assert "D-Base status" in result
        assert "Records:" in result


def test_process_query_clarify_intent():
    """When LLM returns clarify intent, the question is echoed."""
    import json

    iris = Iris()
    response = json.dumps(
        {
            "intent": "clarify",
            "question": "Which disease are you interested in?",
        }
    )

    with patch(
        "dargus.models.compat.llm_from_config",
        _fake_backend(response),
    ):
        result = iris.process_query("predict aspirin")
        assert "Which disease are you interested in?" in result


def test_process_query_chat_intent():
    """When LLM returns chat intent, the message is shown."""
    import json

    iris = Iris()
    response = json.dumps(
        {
            "intent": "chat",
            "message": "I can help you predict drug efficacy. Try 'predict aspirin for headache'.",
        }
    )

    with patch(
        "dargus.models.compat.llm_from_config",
        _fake_backend(response),
    ):
        result = iris.process_query("hello")
        assert "I can help you predict" in result


def test_process_query_invalid_json_fallback():
    """Malformed LLM response returns a friendly error."""
    iris = Iris()

    with patch(
        "dargus.models.compat.llm_from_config",
        _fake_backend("not valid json {{{"),
    ):
        result = iris.process_query("blah")
        assert "I had trouble understanding" in result
