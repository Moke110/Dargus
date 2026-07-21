"""bench-full-stack workflow: benchmark against a stripped copy of the global D-Base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dargus.benchmarks.runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def run(
    strip: dict[str, Any],
    split: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run a bench-full-stack benchmark without touching the core D-Base.

    Args:
        strip: Filter dict for extracting matching records from the global D-Base.
        split: Optional split config, e.g. ``{"test_size": 0.2, "random_state": 42}``.
        output_dir: Optional output directory for reports. Defaults to a temp dir.

    Returns:
        Dict with keys ``metrics``, ``predictions``, ``conditions``.
    """
    split_cfg = split or {}
    conditions = [{"name": "default", "split": split_cfg}]
    config: dict[str, Any] = {
        "strip": strip,
        "conditions": conditions,
    }

    runner = BenchmarkRunner(config)
    if output_dir:
        runner.work_dir = Path(output_dir)

    result = runner.run()
    logger.info("bench-full-stack complete: %s", result.get("metrics"))
    return result
