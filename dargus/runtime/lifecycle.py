"""LifecycleManager — startup, shutdown, and workflow orchestration skeleton."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dargus.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages the Dargus runtime lifecycle: startup, run, shutdown.

    Phase E: workflow methods delegate to the hook-orchestrated functions
    in ``dargus.workflows``.
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        self._runtime = runtime

    def startup(self) -> bool:
        """Run health check and mark the runtime as healthy.

        Returns:
            True if the health check passes, False otherwise.
        """
        from dargus.runtime.context import health_check

        ok = health_check(self._runtime)
        self._runtime.healthy = ok
        if ok:
            logger.info("LifecycleManager startup complete — runtime is healthy")
        else:
            logger.warning("LifecycleManager startup: health check failed — runtime NOT healthy")
        return ok

    def shutdown(self) -> None:
        """Mark runtime unhealthy and log shutdown."""
        self._runtime.healthy = False
        logger.info("LifecycleManager shutdown complete")

    def run_predict(self, task_spec: dict) -> dict:
        """Execute a predict workflow via the hook-orchestrated function.

        Args:
            task_spec: Dict with ``workflow``, ``drug_ids``, ``disease_id``,
                optional ``endpoints``, ``max_rounds``, etc.

        Returns:
            PredictResult dict from ``dargus.workflows.predict.run_predict``.
        """
        from dargus.workflows.predict import run_predict

        return run_predict(task_spec)

    def run_ingest(self, task_spec: dict) -> dict:
        """Execute an ingest workflow via the hook-orchestrated function.

        Args:
            task_spec: Dict with ``workflow``, ``source_path``, optional
                ``source_type``, ``max_rounds``, ``require_confirmation``.

        Returns:
            IngestResult dict from ``dargus.workflows.ingest.run_ingest``.
        """
        from dargus.workflows.ingest import run_ingest

        return run_ingest(task_spec)

    def run_benchmark(self, task_spec: dict) -> dict:
        """Execute a benchmark workflow via the hook-orchestrated function.

        Args:
            task_spec: Dict with ``workflow``, ``holdout_ids``, optional
                ``drug_ids``, ``disease_id``, ``endpoints``.

        Returns:
            BenchmarkResult dict from ``dargus.workflows.benchmark.run_benchmark``.
        """
        from dargus.workflows.benchmark import run_benchmark

        return run_benchmark(task_spec)
