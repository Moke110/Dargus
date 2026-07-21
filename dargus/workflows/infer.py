"""Infer workflow: predict efficacy with a mandatory training-pre-confirmation gate.

DEPRECATED: use ``dargus.workflows.predict`` instead.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

from dargus.iris.base import PredictionMatrix
from dargus.workflows.predict import run as run_predict

logger = logging.getLogger(__name__)

warnings.warn(
    "dargus.workflows.infer is deprecated, use dargus.workflows.predict instead",
    DeprecationWarning,
    stacklevel=2,
)


def run(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
    datadir: str | None = None,
    confirm_callback: Callable[[dict[str, Any]], bool] | None = None,
) -> PredictionMatrix | dict[str, Any]:
    """Deprecated: use ``dargus.workflows.predict.run()`` instead."""
    warnings.warn(
        "workflows.infer.run() is deprecated, use workflows.predict.run() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_predict(
        drug_ids=drug_ids,
        disease_id=disease_id,
        endpoints=endpoints or [],
    )
