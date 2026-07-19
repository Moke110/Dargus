"""Benchmark workflow — evaluate prediction performance."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run(config_path: str) -> dict[str, Any]:
    """Run the Benchmark workflow.

    This is a stub that will be fully implemented in a future task.
    """
    logger.warning("Benchmark workflow is not yet implemented (config=%s)", config_path)
    return {
        "status": "not_implemented",
        "config_path": config_path,
        "metrics": {},
    }
