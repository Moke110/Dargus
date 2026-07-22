"""Tests for DargusLLM — unified HTTP-based LLM client."""

import httpx
import pytest

from dargus.llm import DargusLLM, llm_from_config


class TestDargusLLM:
    """Unit tests for DargusLLM.chat() and complete()."""

    def test_chat_sends_openai_compatible_request(self):
        """chat() POSTs to /v1/chat/completions with correct JSON body."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": "hello"}}]}
            )
        )
        llm = DargusLLM(
            model="test-model",
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            http_client=httpx.Client(transport=transport),
        )
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_complete_wraps_prompt_as_user_message(self):
        """complete() wraps a plain string in a single user message."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": "wrapped"}}]}
            )
        )
        llm = DargusLLM(
            model="test-model",
            base_url="https://api.example.com/v1",
            http_client=httpx.Client(transport=transport),
        )
        result = llm.complete("plain prompt")
        assert result == "wrapped"

    def test_no_api_key_omits_auth_header(self):
        """When api_key is None, no Authorization header is sent."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": "no-auth"}}]}
            )
        )
        llm = DargusLLM(
            model="local-model",
            base_url="http://localhost:11434/v1",
            api_key=None,
            http_client=httpx.Client(transport=transport),
        )
        result = llm.chat([{"role": "user", "content": "hi"}])
        assert result == "no-auth"

    def test_http_error_raises(self):
        """Non-2xx responses raise httpx.HTTPStatusError."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "server error"})
        )
        llm = DargusLLM(
            model="bad",
            base_url="http://localhost",
            http_client=httpx.Client(transport=transport),
        )
        with pytest.raises(httpx.HTTPStatusError):
            llm.chat([{"role": "user", "content": "crash"}])


class TestLlmFromConfig:
    """Tests for the llm_from_config factory function."""

    def test_reads_config_from_yaml_and_env(self, tmp_path, monkeypatch):
        """llm_from_config reads model/base_url from config and api_key from env."""
        import yaml

        config = {
            "llm": {
                "provider": "openai_compatible",
                "model": "deepseek-chat",
                "api_key": "$TEST_KEY",
                "base_url": "https://api.deepseek.com/v1",
                "temperature": 0.0,
                "max_tokens": 2048,
            }
        }
        config_path = tmp_path / "dargus_config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("TEST_KEY", "sk-env-key")

        llm = llm_from_config(config)
        assert llm.model == "deepseek-chat"
        assert llm.base_url == "https://api.deepseek.com/v1"
        assert llm.api_key == "sk-env-key"

    def test_local_llm_no_api_key_needed(self):
        """Local LLM configs work without api_key."""
        config = {
            "llm": {
                "provider": "openai_compatible",
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434/v1",
            }
        }
        llm = llm_from_config(config)
        assert llm.model == "llama3.1:8b"
        assert llm.api_key is None

    def test_empty_config_returns_none(self):
        """Empty config returns None (no LLM configured)."""
        assert llm_from_config({}) is None
        assert llm_from_config(None) is None
