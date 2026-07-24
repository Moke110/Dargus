"""Tests for LifecycleManager."""

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

    def test_run_predict_delegates_to_workflow(self):
        """Phase E: LifecycleManager.run_predict delegates to the workflow function."""
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        result = lm.run_predict(
            {
                "workflow": "predict",
                "drug_ids": ["aspirin"],
                "disease_id": "headache",
                "max_rounds": 1,
            }
        )
        assert isinstance(result, dict)
        assert result["workflow"] == "predict"

    def test_run_ingest_delegates_to_workflow(self):
        """Phase E: LifecycleManager.run_ingest delegates to the workflow function."""
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        result = lm.run_ingest({"workflow": "ingest", "source_path": "/data/test", "max_rounds": 1})
        assert isinstance(result, dict)
        assert result["workflow"] == "ingest"

    def test_run_benchmark_delegates_to_workflow(self):
        """Phase E: LifecycleManager.run_benchmark delegates to the workflow function."""
        ctx = RuntimeContext()
        lm = LifecycleManager(ctx)
        result = lm.run_benchmark({"workflow": "benchmark", "holdout_ids": ["h1"], "max_rounds": 1})
        assert isinstance(result, dict)
        assert result["workflow"] == "benchmark"
