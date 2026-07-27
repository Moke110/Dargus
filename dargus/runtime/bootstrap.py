"""Bootstrap — assemble RuntimeContext from configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from dargus.config.paths import get_config_path
from dargus.models.config import EnvSecretsManager, load_model_config
from dargus.models.embedding import EmbeddingModel, SentenceTransformerBackend
from dargus.models.reasoning import LiteLLMBackend, ReasoningLLM
from dargus.models.router import ModelRouter
from dargus.runtime.context import RuntimeContext, health_check

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


def bootstrap(config_path: str | None = None) -> RuntimeContext:
    """Create a fully assembled RuntimeContext from a YAML config file.

    Steps:
    1. Load config from YAML (default: ``dargus_config.yaml`` in cwd).
    2. Parse model config via ``load_model_config()``.
    3. Resolve API keys via ``EnvSecretsManager``.
    4. Create ``LiteLLMBackend`` for reasoning, wrap in ``ReasoningLLM``.
    5. Create ``SentenceTransformerBackend`` for embedding, wrap in ``EmbeddingModel``.
    6. Create ``ModelRouter`` with a single "planner" backend.
    7. Assemble ``RuntimeContext``, run ``health_check()``, set ``healthy = True``.
    8. Return the context.

    Args:
        config_path: Path to a YAML config file. If None, looks for
            ``dargus_config.yaml`` in the current working directory.

    Returns:
        A RuntimeContext ready for use (healthy=True if all resources loaded).
    """
    path = config_path or _default_config_path()
    config = _load_yaml(path)

    secrets = EnvSecretsManager()

    try:
        model_config = load_model_config(config, secrets)
    except KeyError as exc:
        logger.warning("Model config incomplete: %s — creating minimal context", exc)
        ctx = RuntimeContext(config=config)
        return ctx

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
    )
    reasoning_llm = ReasoningLLM(backend=reasoning_backend)

    # Embedding backend + model (only if embedding config is provided)
    embedding_model = None
    if model_config.embedding_model:
        embedding_model = EmbeddingModel(
            backend=SentenceTransformerBackend(model_name=model_config.embedding_model)
        )

    # ModelRouter — routes reasoning calls by agent phase
    model_router = ModelRouter(backends={"planner": reasoning_backend})

    # Assemble context
    ctx = RuntimeContext(
        config=config,
        reasoning_llm=reasoning_llm,
        embedding_model=embedding_model,
        model_router=model_router,
    )

    if health_check(ctx):
        ctx.healthy = True

    return ctx
