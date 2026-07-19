"""Infer workflow: predict efficacy with a mandatory training-pre-confirmation gate."""

from __future__ import annotations

import logging
from typing import Any, Callable

from dargus.iris.base import PredictionMatrix
from dargus.iris.commander import Iris
from dargus.workflows.train import run as run_train

logger = logging.getLogger(__name__)


def run(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
    datadir: str | None = None,
    confirm_callback: Callable[[dict[str, Any]], bool] | None = None,
) -> PredictionMatrix | dict[str, Any]:
    """Run the Infer workflow.

    The first confirmation callback asks whether to train on ``datadir`` before
    predicting. If the user rejects, inference continues on the existing global
    D-Base. If the user accepts and ``datadir`` is provided, Train runs first.
    """
    if confirm_callback is None:
        confirm_callback = _default_confirm

    train_plan = {
        "kind": "training_pre_confirmation",
        "datadir": datadir,
        "message": "Add new data for supplemental training before inference? [y/N]",
    }
    if not confirm_callback(train_plan):
        return {"aborted": True, "reason": "User declined training-pre-confirmation gate."}

    if datadir is not None:
        train_report = run_train(datadir)
        logger.info("Training report: %s", train_report)

    iris = Iris()
    return iris.infer(
        drug_ids=drug_ids,
        disease_id=disease_id,
        endpoints=endpoints,
        confirm_callback=confirm_callback,
    )


def _default_confirm(_plan: dict[str, Any]) -> bool:
    return False
