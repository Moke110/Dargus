from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    """Protocol for LLM completions used by Iris-*."""

    def complete(self, prompt: str, **kwargs: Any) -> str: ...


class MockLLMBackend:
    """Default backend for tests and offline runs. Returns empty JSON."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return json.dumps({})


class LiteLLMBackend:
    """Provider-agnostic LLM backend via litellm."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        extra: dict[str, Any] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = extra or {}

    def complete(self, prompt: str, **kwargs: Any) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "litellm is required for LiteLLMBackend. "
                "Install with: pip install 'dargus[llm]'"
            ) from exc

        messages = [{"role": "user", "content": prompt}]
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra,
            **kwargs,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        logger.debug("Calling litellm completion with model=%s", self.model)
        response = litellm.completion(**params)
        content = response.choices[0].message.content
        return content if content is not None else json.dumps({})


def llm_backend_from_config(config: dict[str, Any] | None = None) -> LLMBackend:
    """Build an LLM backend from a Dargus config dict.

    Config shape under `llm`:
      provider: mock | litellm
      model: gpt-4o-mini
      api_key: null or "$OPENAI_API_KEY" to read from environment
      base_url: null
      temperature: 0.0
      max_tokens: 2048
      extra: {}
    """
    cfg = config or {}
    llm_cfg = cfg.get("llm") or {}
    provider = llm_cfg.get("provider", "mock")

    if provider == "mock":
        return MockLLMBackend()

    if provider == "litellm":
        model = llm_cfg.get("model")
        if not model:
            raise ValueError("llm.model is required when provider='litellm'")
        api_key = llm_cfg.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            api_key = os.environ.get(api_key[1:])
        return LiteLLMBackend(
            model=model,
            api_key=api_key,
            base_url=llm_cfg.get("base_url"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
            extra=llm_cfg.get("extra", {}),
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
