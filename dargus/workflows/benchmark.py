"""Benchmark workflow: run a config-driven benchmark without modifying global D-Base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from dargus.benchmarks.runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def run(config_path: str) -> dict[str, Any]:
    """Load benchmark config and execute the runner."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    runner = BenchmarkRunner(config)
    return runner.run()
