"""Benchmark workflow — measure predict accuracy against holdout ground truth.

Marks holdout records, runs predict on the remaining active records, compares
predictions to ground truth, and restores holdout state.
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
    2. Mark holdout-test records (exclude from query scope)
    3. Call run_predict using active records
    4. Compare predictions to holdout ground truth
    5. Compute metrics: accuracy, precision, recall, f1
    6. Restore holdout records to active state
    7. Report BenchmarkResult with metrics

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"benchmark"``),
            ``holdout_ids``, optional ``metric``, ``max_rounds``,
            ``require_confirmation``.

    Returns:
        BenchmarkResult dict with keys: ``workflow``, ``status``,
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``n_test``,
        ``session``.
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

    # ---- Mark holdout records -------------------------------------------------
    holdout_ids = task_spec.get("holdout_ids", [])
    logger.info("Marking %d records as holdout-test", len(holdout_ids))
    _mark_holdout(holdout_ids)

    # ---- Run predict workflow on active records --------------------------------
    predict_spec = {
        "workflow": "predict",
        "drug_ids": task_spec.get("drug_ids", []),
        "disease_id": task_spec.get("disease_id", "unknown"),
        "endpoints": task_spec.get("endpoints", []),
        "max_rounds": task_spec.get("max_rounds", 5),
    }
    predict_result = _run_predict_standalone(predict_spec)

    # ---- Compare predictions to ground truth ----------------------------------
    ground_truth = _load_ground_truth(holdout_ids)
    metrics = _compute_metrics(predict_result.get("report", {}), ground_truth)

    # ---- Restore holdout state ------------------------------------------------
    _restore_holdout(holdout_ids)

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
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mark_holdout(holdout_ids: list[str]) -> None:
    """Mark records as holdout-test (excluded from query scope)."""
    # Stub: in a real implementation this would update D-Base metadata
    logger.info("Holdout marked: %s", holdout_ids)


def _restore_holdout(holdout_ids: list[str]) -> None:
    """Restore holdout records to active state."""
    logger.info("Holdout restored: %s", holdout_ids)


def _load_ground_truth(holdout_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load ground truth labels for holdout records (stub)."""
    # Stub: return synthetic ground truth
    return {
        rid: {"actual_efficacy": 0.5 + (hash(rid) % 50) / 100.0, "outcome": "positive"}
        for rid in holdout_ids
    }


def _run_predict_standalone(predict_spec: dict[str, Any]) -> dict[str, Any]:
    """Run predict workflow in a self-contained stub environment.

    Uses the stubs built into ``run_predict`` (no RuntimeContext needed).
    """
    from dargus.workflows.predict import run_predict

    return run_predict(predict_spec)


def _compute_metrics(
    report: dict[str, Any], ground_truth: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Compute accuracy, precision, recall, f1 against ground truth.

    Stub implementation: returns deterministic metrics based on report content.
    """
    n = len(ground_truth)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Stub: derive mock metrics from hash of ground truth keys
    base = sum(hash(k) for k in ground_truth) % 100
    accuracy = 0.5 + base / 200.0
    precision = 0.45 + base / 220.0
    recall = 0.5 + base / 200.0
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
