"""Model configuration loader and secrets management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class SecretsManager(Protocol):
    """Protocol for resolving secret keys (API keys, passwords, etc.)."""

    def get_secret(self, key: str) -> str:
        """Return the secret value for the given key. Raises KeyError if not found."""
        ...


class EnvSecretsManager:
    """SecretsManager that reads from environment variables."""

    def get_secret(self, key: str) -> str:
        """Read a secret from os.environ. Raises KeyError if the variable is not set."""
        value = os.environ.get(key)
        if value is None:
            raise KeyError(f"Environment variable '{key}' is not set")
        return value


@dataclass
class ModelConfig:
    """Parsed model configuration for reasoning and embedding backends.

    All provider/model fields are required; temperature/max_tokens have defaults.
    API key env var names are stored for later resolution via SecretsManager.
    """

    # --- reasoning ---
    reasoning_provider: str
    reasoning_model: str
    reasoning_temperature: float = 0.0
    reasoning_max_tokens: int = 4096
    reasoning_api_key_env: str = ""

    # --- embedding ---
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key_env: str = ""


def load_model_config(config_dict: dict | None, secrets: SecretsManager) -> ModelConfig:
    """Parse and validate model configuration from a config dict.

    Expected shape (under ``models`` key):
        models:
          reasoning:
            provider: anthropic
            model: claude-sonnet-4
            temperature: 0.0
            max_tokens: 4096
            api_key_env: ANTHROPIC_API_KEY
          embedding:
            provider: openai
            model: text-embedding-3-small
            api_key_env: OPENAI_API_KEY

    Args:
        config_dict: The full config dict (top-level keys like ``models``).
        secrets: A SecretsManager for resolving API keys.

    Returns:
        ModelConfig with resolved values.

    Raises:
        KeyError: If a required config key is missing or an env var is not set.
    """
    models_cfg = config_dict.get("models") if config_dict else {}

    reasoning_cfg = models_cfg.get("reasoning", {}) if models_cfg else {}
    embedding_cfg = models_cfg.get("embedding", {}) if models_cfg else {}

    reasoning_provider = reasoning_cfg.get("provider", "")
    reasoning_model = reasoning_cfg.get("model", "")
    if not reasoning_provider:
        raise KeyError("models.reasoning.provider is required")
    if not reasoning_model:
        raise KeyError("models.reasoning.model is required")

    reasoning_api_key_env = reasoning_cfg.get("api_key_env", "")
    if reasoning_api_key_env:
        secrets.get_secret(reasoning_api_key_env)

    embedding_provider = embedding_cfg.get("provider", "")
    embedding_model = embedding_cfg.get("model", "")
    embedding_api_key_env = embedding_cfg.get("api_key_env", "")
    if embedding_api_key_env:
        secrets.get_secret(embedding_api_key_env)

    return ModelConfig(
        reasoning_provider=reasoning_provider,
        reasoning_model=reasoning_model,
        reasoning_temperature=float(reasoning_cfg.get("temperature", 0.0)),
        reasoning_max_tokens=int(reasoning_cfg.get("max_tokens", 4096)),
        reasoning_api_key_env=reasoning_api_key_env,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key_env=embedding_api_key_env,
    )
