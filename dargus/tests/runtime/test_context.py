"""Tests for DargusRuntime dataclass, health flag, and health_check."""

import pytest

from dargus.runtime.context import DargusRuntime, health_check
from dargus.tools.cache import ToolCache


class TestDargusRuntime:
    """Tests for DargusRuntime creation and defaults."""

    def test_default_construction(self):
        rt = DargusRuntime()
        assert rt.config == {}
        assert rt.reasoning_llm is None
        assert rt.embedding_model is None
        assert rt.tool_registry is None
        assert rt.skill_registry is None
        assert rt.dbase_store is None
        assert rt.hook_registry is None

    def test_starts_healthy(self):
        """The runtime starts healthy (design/3_runtime.md health flag)."""
        rt = DargusRuntime()
        assert rt.healthy is True
        assert rt.unhealthy_reason is None

    def test_factory_and_tool_cache_auto_wired(self):
        rt = DargusRuntime()
        assert rt.agent_factory is not None
        assert rt.agent_factory._runtime is rt
        assert isinstance(rt.tool_cache, ToolCache)

    def test_injected_factory_and_cache_kept(self):
        factory = object()
        cache = object()
        rt = DargusRuntime(agent_factory=factory, tool_cache=cache)
        assert rt.agent_factory is factory
        assert rt.tool_cache is cache

    def test_custom_config(self):
        config = {"key": "value"}
        rt = DargusRuntime(config=config)
        assert rt.config == config

    def test_set_reasoning_llm(self):
        rt = DargusRuntime()
        fake_llm = object()
        rt.reasoning_llm = fake_llm  # type: ignore[assignment]
        assert rt.reasoning_llm is fake_llm

    def test_set_embedding_model(self):
        rt = DargusRuntime()
        fake_emb = object()
        rt.embedding_model = fake_emb  # type: ignore[assignment]
        assert rt.embedding_model is fake_emb


class TestHealthFlag:
    """Health flag semantics: starts healthy, unhealthy only on failure."""

    def test_mark_unhealthy_records_reason(self):
        rt = DargusRuntime()
        rt.mark_unhealthy("D-Base inaccessible")
        assert rt.healthy is False
        assert rt.unhealthy_reason == "D-Base inaccessible"

    def test_ensure_healthy_passes_when_healthy(self):
        DargusRuntime().ensure_healthy()  # must not raise

    def test_ensure_healthy_raises_when_unhealthy(self):
        rt = DargusRuntime()
        rt.mark_unhealthy("model unavailable")
        with pytest.raises(RuntimeError, match="model unavailable"):
            rt.ensure_healthy()

    def test_shutdown_closes_tool_cache_and_marks_unhealthy(self):
        rt = DargusRuntime()
        rt.tool_cache.put("heavy", object())
        rt.shutdown()
        assert rt.healthy is False
        with pytest.raises(RuntimeError, match="closed"):
            rt.tool_cache.get("heavy")


class TestHealthCheck:
    """Tests for health_check() — presence of both model dependencies."""

    def test_healthy_when_both_models_present(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        rt.embedding_model = object()  # type: ignore[assignment]
        assert health_check(rt) is True

    def test_unhealthy_when_reasoning_llm_missing(self):
        rt = DargusRuntime()
        rt.embedding_model = object()  # type: ignore[assignment]
        assert health_check(rt) is False

    def test_unhealthy_when_embedding_model_missing(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        assert health_check(rt) is False

    def test_unhealthy_when_both_missing(self):
        assert health_check(DargusRuntime()) is False
