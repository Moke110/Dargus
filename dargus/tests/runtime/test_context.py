"""Tests for RuntimeContext dataclass and health_check."""

from dargus.runtime.context import RuntimeContext, health_check


class TestRuntimeContext:
    """Tests for RuntimeContext creation and defaults."""

    def test_default_construction(self):
        ctx = RuntimeContext()
        assert ctx.config == {}
        assert ctx.reasoning_llm is None
        assert ctx.embedding_model is None
        assert ctx.tool_registry is None
        assert ctx.skill_registry is None
        assert ctx.knowledge_retrievers == {}
        assert ctx.dbase_manager is None
        assert ctx.hook_registry is None
        assert ctx.healthy is False

    def test_custom_config(self):
        config = {"key": "value"}
        ctx = RuntimeContext(config=config)
        assert ctx.config == config

    def test_set_reasoning_llm(self):
        ctx = RuntimeContext()
        fake_llm = object()
        ctx.reasoning_llm = fake_llm  # type: ignore[assignment]
        assert ctx.reasoning_llm is fake_llm

    def test_set_embedding_model(self):
        ctx = RuntimeContext()
        fake_emb = object()
        ctx.embedding_model = fake_emb  # type: ignore[assignment]
        assert ctx.embedding_model is fake_emb


class TestHealthCheck:
    """Tests for health_check()."""

    def test_healthy_when_both_models_present(self):
        ctx = RuntimeContext()
        ctx.reasoning_llm = object()  # type: ignore[assignment]
        ctx.embedding_model = object()  # type: ignore[assignment]
        assert health_check(ctx) is True

    def test_unhealthy_when_reasoning_llm_missing(self):
        ctx = RuntimeContext()
        ctx.embedding_model = object()  # type: ignore[assignment]
        assert health_check(ctx) is False

    def test_unhealthy_when_embedding_model_missing(self):
        ctx = RuntimeContext()
        ctx.reasoning_llm = object()  # type: ignore[assignment]
        assert health_check(ctx) is False

    def test_unhealthy_when_both_missing(self):
        ctx = RuntimeContext()
        assert health_check(ctx) is False
