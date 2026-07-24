"""Tests for LifecycleManager."""

import pytest

from dargus.runtime.context import RuntimeContext
from dargus.runtime.lifecycle import LifecycleManager


class TestLifecycleManager:
    """Tests for LifecycleManager startup/shutdown flow and stub methods."""

    def test_constructor_stores_runtime(self):
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        assert lm._runtime is ctx

    def test_startup_sets_healthy_when_models_present(self):
        ctx = RuntimeContext()
        ctx.reasoning_llm = object()  # type: ignore[assignment]
        ctx.embedding_model = object()  # type: ignore[assignment]

        lm = LifecycleManager(ctx)
        result = lm.startup()

        assert result is True
        assert ctx.healthy is True

    def test_startup_sets_unhealthy_when_models_missing(self):
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        result = lm.startup()

        assert result is False
        assert ctx.healthy is False

    def test_shutdown_sets_unhealthy(self):
        ctx = RuntimeContext()
        ctx.healthy = True
        lm = LifecycleManager(ctx)
        lm.shutdown()

        assert ctx.healthy is False

    def test_startup_then_shutdown_cycle(self):
        ctx = RuntimeContext()
        ctx.reasoning_llm = object()  # type: ignore[assignment]
        ctx.embedding_model = object()  # type: ignore[assignment]

        lm = LifecycleManager(ctx)

        assert lm.startup() is True
        assert ctx.healthy is True

        lm.shutdown()
        assert ctx.healthy is False

    def test_run_predict_raises_not_implemented(self):
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        with pytest.raises(NotImplementedError, match="run_predict not implemented yet"):
            lm.run_predict({"drug": "aspirin"})

    def test_run_ingest_raises_not_implemented(self):
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        with pytest.raises(NotImplementedError, match="run_ingest not implemented yet"):
            lm.run_ingest({"file": "data.csv"})

    def test_run_benchmark_raises_not_implemented(self):
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        with pytest.raises(NotImplementedError, match="run_benchmark not implemented yet"):
            lm.run_benchmark({"holdout": 0.2})
