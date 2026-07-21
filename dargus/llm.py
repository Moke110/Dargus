"""DargusLLM — unified OpenAI-compatible HTTP client for online and local LLMs."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DargusLLM:
    """Single-protocol LLM client via POST /v1/chat/completions.

    Works with any OpenAI-compatible endpoint: OpenAI, DeepSeek, Groq,
    Ollama, vLLM, llama.cpp server, etc.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        http_client: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = http_client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages to /v1/chat/completions, return assistant content."""
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        logger.debug("DargusLLM request to %s (model=%s)", url, self.model)
        response = self.client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def complete(self, prompt: str) -> str:
        """Convenience: wrap a plain prompt as a single user message."""
        return self.chat([{"role": "user", "content": prompt}])


def check_llm_connection(llm: DargusLLM) -> dict[str, Any]:
    """Send a minimal test request to verify LLM connectivity.

    Returns:
        {"ok": True, "model": "...", "latency_ms": 237}
        {"ok": False, "model": "...", "error": "..."}
    """
    import time

    t0 = time.monotonic()
    try:
        response = llm.chat([{"role": "user", "content": "Reply with just: OK"}])
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {"ok": True, "model": llm.model, "latency_ms": elapsed_ms}
    except Exception as exc:
        return {"ok": False, "model": llm.model, "error": str(exc)}


def llm_from_config(config: dict[str, Any] | None = None) -> DargusLLM | None:
    """Build a DargusLLM from a Dargus config dict.

    If config is None or empty, returns None (no LLM configured).

    Config shape under ``llm``:
      provider: openai_compatible | mock
      model: deepseek-chat
      api_key: "$DARGUS_LLM_API_KEY" or null
      base_url: https://api.deepseek.com/v1
      temperature: 0.0
      max_tokens: 2048
    """
    cfg = config or {}
    llm_cfg = cfg.get("llm") or {}
    provider = llm_cfg.get("provider", "")

    if not provider or provider == "mock":
        return None

    model = llm_cfg.get("model")
    if not model:
        raise ValueError("llm.model is required when provider is set")

    def _resolve(value: Any) -> str | None:
        if isinstance(value, str) and value.startswith("$"):
            return os.environ.get(value[1:])
        return value

    api_key = _resolve(llm_cfg.get("api_key"))
    base_url = _resolve(llm_cfg.get("base_url", ""))
    if not base_url:
        return None

    return DargusLLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=int(llm_cfg.get("max_tokens", 2048)),
    )
