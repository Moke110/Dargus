"""Bootstrap — assemble a DargusRuntime from configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from dargus.config.paths import get_config_path
from dargus.models.config import EnvSecretsManager, load_model_config
from dargus.models.embedding import EmbeddingModel, SentenceTransformerBackend
from dargus.models.reasoning import LiteLLMBackend, ReasoningLLM, ReasoningOptions
from dargus.runtime.context import DargusRuntime

logger = logging.getLogger(__name__)


def _default_config_path() -> str:
    """Return the default config file path (unified config path)."""
    return str(get_config_path())


def _load_yaml(path: str) -> dict:
    """Load a YAML config file, returning an empty dict if the file is missing."""
    p = Path(path)
    if not p.exists():
        logger.debug("Config file %s not found — using empty config", path)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def bootstrap(config_path: str | None = None) -> DargusRuntime:
    """Create a fully assembled DargusRuntime from a YAML config file.

    Steps:
    1. Load config from YAML (default: the unified Dargus config path).
    2. Parse model config via ``load_model_config()``.
    3. Resolve API keys via ``EnvSecretsManager``.
    4. Create ``LiteLLMBackend`` for reasoning, wrap in ``ReasoningLLM``.
    5. Create ``SentenceTransformerBackend`` for embedding, wrap in
       ``EmbeddingModel``.
    6. Assemble the ``DargusRuntime``. The runtime starts healthy; a
       missing model config marks it unhealthy at entry (the reasoning LLM
       is a hard dependency — no model router exists in v1.0.0).

    Args:
        config_path: Path to a YAML config file. If None, the unified
            Dargus config path is used.

    Returns:
        A DargusRuntime (healthy unless a hard dependency is unavailable).
    """
    path = config_path or _default_config_path()
    config = _load_yaml(path)

    secrets = EnvSecretsManager()

    try:
        model_config = load_model_config(config, secrets)
    except KeyError as exc:
        logger.warning("Model config incomplete: %s — creating minimal runtime", exc)
        return DargusRuntime(config=config)

    # Reasoning backend + LLM
    reasoning_api_key = ""
    if model_config.reasoning_api_key_env:
        try:
            reasoning_api_key = secrets.get_secret(model_config.reasoning_api_key_env)
        except KeyError:
            logger.warning(
                "API key env var '%s' not set — reasoning backend may fail",
                model_config.reasoning_api_key_env,
            )

    reasoning_backend = LiteLLMBackend(
        provider=model_config.reasoning_provider,
        model=model_config.reasoning_model,
        api_key=reasoning_api_key,
        base_url=model_config.reasoning_base_url or None,
    )
    reasoning_llm = ReasoningLLM(
        backend=reasoning_backend,
        default_options=ReasoningOptions(
            temperature=model_config.reasoning_temperature,
            max_tokens=model_config.reasoning_max_tokens,
        ),
    )

    # Embedding backend + model (only if embedding config is provided)
    embedding_model = None
    if model_config.embedding_model:
        embedding_model = EmbeddingModel(
            backend=SentenceTransformerBackend(model_name=model_config.embedding_model)
        )

    # The runtime starts healthy; it turns unhealthy only on an
    # unrecoverable failure observed at run time (see mark_unhealthy()).
    return DargusRuntime(
        config=config,
        reasoning_llm=reasoning_llm,
        embedding_model=embedding_model,
    )
