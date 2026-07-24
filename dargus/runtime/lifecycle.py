"""LifecycleManager — startup, shutdown, and workflow orchestration skeleton."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dargus.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages the Dargus runtime lifecycle: startup, run, shutdown.

    In Phase A all workflow methods are stubs.  Real implementations arrive
    in Phase E.
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

    def run_predict(self, task_spec: dict):
        """Stub — execute a predict workflow."""
        raise NotImplementedError("run_predict not implemented yet")

    def run_ingest(self, task_spec: dict):
        """Stub — execute an ingest workflow."""
        raise NotImplementedError("run_predict not implemented yet")

    def run_benchmark(self, task_spec: dict):
        """Stub — execute a benchmark workflow."""
        raise NotImplementedError("run_predict not implemented yet")
