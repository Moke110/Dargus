"""Predict workflow — thin wrapper around Iris.predict() for the CLI."""

from __future__ import annotations

import logging
from typing import Any

from dargus.iris.commander import Iris

logger = logging.getLogger(__name__)


def run(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
    max_rounds: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the Predict workflow.

    This is a thin shell around ``Iris.predict()``. The CLI calls this
    function so that workflows remain the entry point layer.
    """
    iris = Iris()
    return iris.predict(
        drug_ids=drug_ids,
        disease_id=disease_id,
        endpoints=endpoints or [],
        max_rounds=max_rounds,
    )
