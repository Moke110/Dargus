"""Dargus model layer — ReasoningLLM, EmbeddingModel, and configuration."""

from dargus.models.config import EnvSecretsManager, ModelConfig, SecretsManager, load_model_config
from dargus.models.conversation import (
    Conversation,
    ConvMessage,
    ToolCall,
    ToolResult,
)
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

__all__ = [
    "SecretsManager",
    "EnvSecretsManager",
    "ModelConfig",
    "load_model_config",
    "Message",
    "Conversation",
    "ConvMessage",
    "ToolCall",
    "ToolResult",
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
]
