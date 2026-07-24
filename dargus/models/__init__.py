"""Dargus model layer — ReasoningLLM, EmbeddingModel, ModelRouter, and configuration."""

from dargus.models.config import EnvSecretsManager, ModelConfig, SecretsManager, load_model_config
from dargus.models.embedding import (
    Embedding,
    EmbeddingBackend,
    EmbeddingModel,
    SentenceTransformerBackend,
)
from dargus.models.reasoning import (
    LiteLLMBackend,
    LLMResponse,
    LLMUsage,
    Message,
    ReasoningBackend,
    ReasoningLLM,
    ReasoningOptions,
)
from dargus.models.router import ModelRouter

__all__ = [
    "SecretsManager",
    "EnvSecretsManager",
    "ModelConfig",
    "load_model_config",
    "Message",
    "ReasoningOptions",
    "LLMUsage",
    "LLMResponse",
    "ReasoningBackend",
    "LiteLLMBackend",
    "ReasoningLLM",
    "Embedding",
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "EmbeddingModel",
    "ModelRouter",
]
