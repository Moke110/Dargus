"""Tests for model configuration loading and SecretsManager."""

import pytest

from dargus.models.config import (
    EnvSecretsManager,
    ModelConfig,
    load_model_config,
)


class TestEnvSecretsManager:
    """Tests for EnvSecretsManager."""

    def test_get_secret_returns_env_value(self, monkeypatch):
        """get_secret() returns the value from os.environ."""
        monkeypatch.setenv("TEST_KEY", "test-value")
        sm = EnvSecretsManager()
        assert sm.get_secret("TEST_KEY") == "test-value"

    def test_get_secret_raises_keyerror_when_missing(self):
        """get_secret() raises KeyError if the env var is not set."""
        sm = EnvSecretsManager()
        with pytest.raises(KeyError, match="NOT_SET_VAR"):
            sm.get_secret("NOT_SET_VAR")


class TestLoadModelConfig:
    """Tests for load_model_config()."""

    def test_valid_config(self, monkeypatch):
        """A valid config dict produces a fully populated ModelConfig."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        config_dict = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "temperature": 0.2,
                    "max_tokens": 8192,
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key_env": "OPENAI_API_KEY",
                },
            }
        }

        mc = load_model_config(config_dict, EnvSecretsManager())

        assert mc.reasoning_provider == "anthropic"
        assert mc.reasoning_model == "claude-sonnet-4"
        assert mc.reasoning_temperature == 0.2
        assert mc.reasoning_max_tokens == 8192
        assert mc.reasoning_api_key_env == "ANTHROPIC_API_KEY"
        assert mc.embedding_provider == "openai"
        assert mc.embedding_model == "text-embedding-3-small"
        assert mc.embedding_api_key_env == "OPENAI_API_KEY"

    def test_defaults_when_optional_fields_missing(self, monkeypatch):
        """Optional fields (temperature, max_tokens) use defaults when not present."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        config_dict = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            }
        }

        mc = load_model_config(config_dict, EnvSecretsManager())

        assert mc.reasoning_temperature == 0.0
        assert mc.reasoning_max_tokens == 4096

    def test_missing_reasoning_provider_raises(self):
        """Missing models.reasoning.provider raises KeyError."""
        config_dict = {"models": {"reasoning": {"model": "claude-sonnet-4"}}}
        with pytest.raises(KeyError, match="reasoning.provider"):
            load_model_config(config_dict, EnvSecretsManager())

    def test_missing_reasoning_model_raises(self):
        """Missing models.reasoning.model raises KeyError."""
        config_dict = {"models": {"reasoning": {"provider": "anthropic"}}}
        with pytest.raises(KeyError, match="reasoning.model"):
            load_model_config(config_dict, EnvSecretsManager())

    def test_missing_api_key_env_var_raises(self):
        """If api_key_env is set but the env var is missing, KeyError is raised."""
        config_dict = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "api_key_env": "MISSING_KEY",
                },
            }
        }
        with pytest.raises(KeyError, match="MISSING_KEY"):
            load_model_config(config_dict, EnvSecretsManager())

    def test_empty_config_dict_returns_defaults_and_raises(self):
        """An empty or missing 'models' section raises KeyError for required fields."""
        with pytest.raises(KeyError):
            load_model_config({}, EnvSecretsManager())

    def test_none_config_treated_as_empty(self):
        """None config is treated as empty dict."""
        with pytest.raises(KeyError):
            load_model_config(None, EnvSecretsManager())  # type: ignore[arg-type]

    def test_embedding_config_without_api_key(self):
        """Embedding config without api_key_env works fine."""
        config_dict = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                },
                "embedding": {
                    "provider": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2",
                },
            }
        }
        mc = load_model_config(config_dict, EnvSecretsManager())
        assert mc.embedding_provider == "sentence_transformers"
        assert mc.embedding_model == "all-MiniLM-L6-v2"
        assert mc.embedding_api_key_env == ""

    def test_modelconfig_dataclass_defaults(self):
        """ModelConfig dataclass fields have correct defaults."""
        mc = ModelConfig(
            reasoning_provider="test",
            reasoning_model="test-model",
        )
        assert mc.reasoning_temperature == 0.0
        assert mc.reasoning_max_tokens == 4096
        assert mc.reasoning_api_key_env == ""
        assert mc.embedding_provider == ""
        assert mc.embedding_model == ""
        assert mc.embedding_api_key_env == ""
