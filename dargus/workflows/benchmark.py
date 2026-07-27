"""Benchmark workflow — measure predict accuracy against holdout ground truth.

Design (design/7_workflows.md §Benchmark):
- Matching records are temporarily marked ``holdout-test`` via the status
  sidecar.
- Predict reads only ``active`` records, so holdout records cannot leak
  into inference.
- Predictions are compared against the held-out ground truth (the y-values
  of the held-out records themselves).
- After Benchmark finishes, all holdout records are restored to ``active``
  (also on error).
- No temporary D-Base is created; zero matched holdout records aborts.
"""

from __future__ import annotations

import logging
from typing import Any

from dargus.runtime.hooks import (
    HookContext,
    HookPoint,
    HookRegistry,
    ReportValidationHook,
    ResultReportHook,
    SafetyNetHook,
    SessionInitHook,
    SkeletonContextHook,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmark(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Execute benchmark workflow with Hook enforcement.

    1. SESSION_START: SessionInitHook creates BenchmarkSession
    2. Resolve holdout records (explicit ``holdout_ids`` or a ``holdout``
       filter dict) and flip them to ``holdout-test`` in the status sidecar
    3. Run predict on the remaining active records
    4. Compare predictions to the held-out ground truth
    5. Compute metrics: accuracy, precision, recall, f1
    6. Restore holdout records to ``active``
    7. Report BenchmarkResult with metrics

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"benchmark"``),
            ``holdout_ids`` (explicit evidence_ids) or ``holdout`` (filter
            dict with optional ``drug_ids`` / ``disease_id`` / ``y_type`` /
            ``level``), plus predict inputs ``drug_ids`` / ``disease_id`` /
            ``endpoints``, optional ``max_rounds``, ``require_confirmation``.

    Returns:
        BenchmarkResult dict with keys: ``workflow``, ``status``,
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``n_test``,
        ``session``.

    Raises:
        ValueError: when the holdout selection matches zero records.
    """
    task_spec.setdefault("workflow", "benchmark")
    max_rounds = int(task_spec.get("max_rounds", 5))

    # ---- Build hook registry --------------------------------------------------
    hooks = HookRegistry()
    hooks.register(HookPoint.SESSION_START, SessionInitHook())
    hooks.register(HookPoint.PERCEIVE_START, SkeletonContextHook(max_rounds=max_rounds))
    hooks.register(HookPoint.ROUND_END, SafetyNetHook(max_rounds=max_rounds, session_timeout=600.0))
    hooks.register(HookPoint.SESSION_END, ReportValidationHook())
    hooks.register(HookPoint.SESSION_END, ResultReportHook())

    # ---- Create initial context ------------------------------------------------
    ctx = HookContext(runtime=None, task_spec=task_spec)

    # ---- SESSION_START hooks --------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_START, ctx)

    # ---- Mark holdout records in the status sidecar ---------------------------
    manager = _manager()
    holdout_records = _resolve_holdout_records(manager, task_spec)
    holdout_ids = [r["evidence_id"] for r in holdout_records]
    if not holdout_ids:
        raise ValueError(
            "Benchmark holdout selection matched zero records — aborting. "
            "Provide holdout_ids or a holdout filter that matches D-Base records."
        )
    logger.info("Marking %d records as holdout-test", len(holdout_ids))
    _mark_holdout(manager, holdout_ids)

    try:
        # ---- Run predict on active records only --------------------------------
        predict_spec = {
            "workflow": "predict",
            "drug_ids": task_spec.get("drug_ids", []),
            "disease_id": task_spec.get("disease_id", "unknown"),
            "endpoints": task_spec.get("endpoints", []),
            "max_rounds": task_spec.get("max_rounds", 5),
        }
        predict_result = _run_predict_standalone(predict_spec)

        # ---- Compare predictions to held-out ground truth ---------------------
        ground_truth = _load_ground_truth(holdout_records)
        metrics = _compute_metrics(predict_result, ground_truth)
    finally:
        # ---- Restore holdout state (even on failure) --------------------------
        _restore_holdout(manager, holdout_ids)

    # ---- Build FinalReport ----------------------------------------------------
    final_report: dict[str, Any] = {
        "metrics": metrics,
        "n_test": len(holdout_ids),
        "holdout_ids": holdout_ids,
        "predict_status": predict_result.get("status"),
    }
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks ----------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate -----------------------------------------------
    _user_confirmation_gate(ctx, task_spec)

    # ---- Return result --------------------------------------------------------
    return {
        "workflow": "benchmark",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_test": len(holdout_ids),
        "report": final_report,
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _manager() -> Any:
    """DBaseManager over the global working D-Base."""
    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    return DBaseManager(DBase.global_instance())


def _resolve_holdout_records(manager: Any, task_spec: dict[str, Any]) -> list[dict]:
    """Resolve the holdout set: explicit ids, or a filter dict.

    Explicit ``holdout_ids`` win. Otherwise the ``holdout`` filter (keys:
    ``drug_ids`` → x_entity, ``disease_id``, ``y_type``, ``level``) selects
    matching records, optionally capped by ``max_records`` / ``fraction``.
    Only active records are candidates.
    """
    explicit = task_spec.get("holdout_ids") or []
    if explicit:
        records = []
        for eid in explicit:
            record = manager.read_record(eid)
            if record is not None:
                records.append(record)
        return records

    holdout_filter = task_spec.get("holdout") or {}
    drug_ids = holdout_filter.get("drug_ids") or task_spec.get("drug_ids") or []
    candidates = manager.read_records(
        x_entity=drug_ids[0] if len(drug_ids) == 1 else None,
        disease_id=holdout_filter.get("disease_id") or task_spec.get("disease_id"),
        y_type=holdout_filter.get("y_type"),
        level=holdout_filter.get("level"),
        status="active",
    )
    if len(drug_ids) > 1:
        wanted = set(drug_ids)
        candidates = [
            r
            for r in candidates
            if any(v.get("entity_id") in wanted for v in (r.get("x", {}).get("value") or []))
        ]

    max_records = holdout_filter.get("max_records")
    if max_records is not None:
        candidates = candidates[: int(max_records)]
    fraction = holdout_filter.get("fraction")
    if fraction is not None and 0.0 < float(fraction) < 1.0:
        n = max(1, int(len(candidates) * float(fraction)))
        candidates = candidates[:n]
    return candidates


def _mark_holdout(manager: Any, holdout_ids: list[str]) -> None:
    """Mark records as holdout-test (excluded from Predict's active scope)."""
    for eid in holdout_ids:
        manager.update_status(eid, "holdout-test")


def _restore_holdout(manager: Any, holdout_ids: list[str]) -> None:
    """Restore holdout records to active state."""
    for eid in holdout_ids:
        manager.update_status(eid, "active")


def _load_ground_truth(holdout_records: list[dict]) -> dict[str, dict[str, Any]]:
    """Ground truth for a held-out record is its own y readout.

    The primary y value thresholded at 0.5 gives the binary outcome used by
    the classification metrics.
    """
    ground_truth: dict[str, dict[str, Any]] = {}
    for record in holdout_records:
        eid = record.get("evidence_id")
        if not eid:
            continue
        y_values = record.get("y", {}).get("value") or []
        actual = y_values[0] if y_values else 0.0
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            actual = 0.0
        ground_truth[eid] = {
            "actual_efficacy": actual,
            "outcome": "positive" if actual >= 0.5 else "negative",
            "y_type": record.get("y", {}).get("type"),
        }
    return ground_truth


def _run_predict_standalone(predict_spec: dict[str, Any]) -> dict[str, Any]:
    """Run the predict workflow (self-contained; no RuntimeContext needed)."""
    from dargus.workflows.predict import run_predict

    return run_predict(predict_spec)


def _compute_metrics(
    predict_result: dict[str, Any], ground_truth: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Compute accuracy, precision, recall, f1 against ground truth.

    The predicted binary outcome is the report's ``efficacy_score``
    thresholded at 0.5 (``None`` — insufficient data — counts as negative).
    The single predicted label is evaluated against every held-out record.
    """
    n = len(ground_truth)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    score = (predict_result.get("report") or {}).get("efficacy_score")
    predicted_positive = score is not None and float(score) >= 0.5

    tp = fp = tn = fn = 0
    for truth in ground_truth.values():
        actual_positive = truth.get("outcome") == "positive"
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _user_confirmation_gate(ctx: HookContext, task_spec: dict[str, Any]) -> None:
    """Stub user confirmation for benchmark — logs and auto-approves."""
    if task_spec.get("require_confirmation"):
        logger.info("User confirmation required — auto-approved (stub)")
    if isinstance(ctx.session, dict):
        ctx.session.setdefault("confirmations", []).append(
            {
                "type": "benchmark_review",
                "action": "auto_approved",
            }
        )
