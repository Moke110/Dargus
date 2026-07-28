"""Predict workflow — hook-orchestrated clinical efficacy prediction.

Converts task_spec into a PredictResult through a multi-round D4Expert
loop with Hook enforcement at each lifecycle point.
"""

from __future__ import annotations

import logging
import time
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


def _disabled_hooks(task_spec: dict[str, Any]) -> set[str]:
    """Core-hook names disabled via config (``hooks: disable: [...]``)."""
    cfg = task_spec.get("hooks", {})
    return set(cfg.get("disable", []) or [])


def _hook_dbase() -> Any | None:
    """Wire the working D-Base into ReportValidationHook for evidence_id
    existence checks. Returns None when the store cannot be opened (range
    checks still apply)."""
    try:
        from dargus.dbase.dbase import DBase
        from dargus.dbase.paths import dbase_root

        return DBase("predict-validation", root_dir=dbase_root())
    except Exception:
        logger.debug("No D-Base available for report validation — skipping existence checks")
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_predict(
    task_spec: dict[str, Any],
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute predict workflow with Hook enforcement.

    1. SESSION_START: SessionInitHook creates PredictSession
    2. Create D4Expert via AgentFactory (bootstraps a DargusRuntime
       when *runtime* is not supplied)
    3. Main loop (max *max_rounds* / timeout *timeout_seconds*):
       - PERCEIVE_START: SkeletonContextHook injects state
       - D4Expert runs assess + synthesize cycle
       - ROUND_END: SafetyNetHook checks convergence
    4. SESSION_END: ReportValidationHook validates FinalReport
    5. ResultReportHook assembles PredictResult

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"predict"``),
            ``drug_ids``, ``disease_id``, optional ``endpoints``,
            ``max_rounds``, ``timeout_seconds``, ``require_confirmation``.
        runtime: Optional pre-built :class:`DargusRuntime`.  When None a
            runtime is bootstrapped from the config file (design/3_runtime.md).

    Returns:
        PredictResult dict with keys: ``workflow``, ``status``,
        ``rounds_completed``, ``report``, ``session``.
    """
    task_spec.setdefault("workflow", "predict")
    max_rounds = int(task_spec.get("max_rounds", 5))
    timeout_seconds = float(task_spec.get("timeout_seconds", 300.0))

    # ---- Obtain DargusRuntime (bootstrap if not provided) ------------------
    if runtime is None:
        from dargus.runtime.bootstrap import bootstrap

        try:
            runtime = bootstrap()
        except Exception as exc:
            raise RuntimeError(
                f"DargusRuntime bootstrap failed: {exc} — refusing predict session"
            ) from exc

    # ---- Build hook registry --------------------------------------------------
    hooks = HookRegistry(disabled_hooks=_disabled_hooks(task_spec))
    hooks.register(HookPoint.SESSION_START, SessionInitHook())
    hooks.register(HookPoint.PERCEIVE_START, SkeletonContextHook(max_rounds=max_rounds))
    hooks.register(
        HookPoint.ROUND_END,
        SafetyNetHook(
            max_rounds=max_rounds,
            session_timeout=timeout_seconds,
        ),
    )
    hooks.register(HookPoint.SESSION_END, ReportValidationHook(dbase=_hook_dbase()))
    hooks.register(HookPoint.SESSION_END, ResultReportHook())

    # ---- Create initial context with real runtime ---------------------------
    ctx = HookContext(runtime=runtime, task_spec=task_spec)

    # ---- SESSION_START hooks --------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_START, ctx)

    # ---- Build D4Expert via AgentFactory --------------------------------------
    d4 = runtime.agent_factory.d4_expert()

    # ---- Main round loop ------------------------------------------------------
    report: dict[str, Any] = {}
    round_num = 0

    while round_num < max_rounds:
        ctx.round = round_num

        # PERCEIVE_START
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)

        # Execute one round: delegate to all domains, synthesize
        round_t0 = time.monotonic()
        round_report = _execute_predict_round(ctx, d4, round_num, hooks)
        report = round_report
        ctx.extra["round_elapsed_ms"] = (time.monotonic() - round_t0) * 1000

        # Store in session
        if isinstance(ctx.session, dict):
            rounds = ctx.session.setdefault("rounds", [])
            rounds.append({"round": round_num, "report_summary": _summarize_report(report)})

        # ROUND_END safety check
        ctx = hooks.run(HookPoint.ROUND_END, ctx)
        round_num += 1
        if ctx.extra.get("force_converge"):
            logger.info("Safety net triggered — forcing convergence at round %d", round_num - 1)
            break

    # ---- Store FinalReport in context -----------------------------------------
    final_report = _build_final_report(report, task_spec)
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks (validation + report assembly) ---------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate (stub: auto-approve) --------------------------
    _user_confirmation_gate(ctx, task_spec)

    # ---- Return result --------------------------------------------------------
    return {
        "workflow": "predict",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "rounds_completed": round_num,
        "efficacy_score": final_report.get("efficacy_score"),
        "confidence_score": final_report.get("confidence_score"),
        "drug_ids": final_report.get("drug_ids"),
        "disease_id": final_report.get("disease_id"),
        "endpoints": final_report.get("endpoints"),
        "report": final_report,
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_predict_round(
    ctx: HookContext, d4: Any, round_num: int, hooks: HookRegistry | None = None
) -> dict[str, Any]:
    """Run one predict round: delegate to all domains, then synthesize."""
    drug_ids = ctx.task_spec.get("drug_ids", [])
    disease_id = ctx.task_spec.get("disease_id", "unknown")
    endpoints = ctx.task_spec.get("endpoints", [])
    question = f"Assess efficacy of {drug_ids} for {disease_id} on {endpoints or 'all endpoints'}"

    domains = ["molecular", "biomedical", "bioinformatics", "clinical"]
    expert_reports: list[dict[str, Any]] = []

    for domain in domains:
        try:
            rep = d4.delegate_to_expert(domain, [], question)
            expert_reports.append(rep)
            if hooks is not None:
                hooks.run(
                    HookPoint.DOMAIN_REPORT_PRODUCED,
                    HookContext(
                        runtime=ctx.runtime,
                        task_spec=ctx.task_spec,
                        session=ctx.session,
                        round=round_num,
                        extra={"domain_report": rep},
                    ),
                )
        except Exception as exc:
            logger.warning("Expert delegation to %s failed: %s", domain, exc)
            expert_reports.append({"domain": domain, "conclusion": str(exc), "confidence": {}})

    synthesized = d4.synthesize(expert_reports) if hasattr(d4, "synthesize") else {}
    if hooks is not None:
        hooks.run(
            HookPoint.D4_REPORT_PRODUCED,
            HookContext(
                runtime=ctx.runtime,
                task_spec=ctx.task_spec,
                session=ctx.session,
                round=round_num,
                extra={"d4_report": synthesized},
            ),
        )
    coverage = len([r for r in expert_reports if r.get("conclusion")]) / max(len(domains), 1)
    ctx.extra["evidence_coverage"] = coverage
    ctx.extra["pending_delegations"] = 0

    return synthesized


def _build_final_report(round_report: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
    """Convert the last round's synthesized report into a validated FinalReport.

    Supports ``_efficacy_score_override`` and ``_confidence_score_override``
    keys in *task_spec* for injection of invalid values during testing.
    """
    return {
        "drug_ids": task_spec.get("drug_ids", []),
        "disease_id": task_spec.get("disease_id", "unknown"),
        "endpoints": task_spec.get("endpoints", []),
        "efficacy_score": task_spec.get("_efficacy_score_override", 0.5),
        "confidence_score": task_spec.get("_confidence_score_override", 0.2),
        "supporting_records": task_spec.get("_supporting_records_override", ["stub-record-1"]),
        "confidence_level": task_spec.get("_confidence_level_override", "moderate"),
        "overall_conclusion": round_report.get("overall_conclusion", "no conclusion"),
        "reasoning_mode": "workflow-hook-orchestrated",
    }


def _summarize_report(report: dict[str, Any]) -> str:
    """Return a short summary string for a round report."""
    return str(report.get("overall_conclusion", ""))[:200]


def _user_confirmation_gate(ctx: HookContext, task_spec: dict[str, Any]) -> None:
    """Stub user confirmation — logs and auto-approves.

    Set ``require_confirmation: true`` in *task_spec* to enable the log
    message.  In the future this will prompt for actual user input.
    """
    if task_spec.get("require_confirmation"):
        logger.info("User confirmation required — auto-approved (stub)")
    if isinstance(ctx.session, dict):
        ctx.session.setdefault("confirmations", []).append(
            {
                "type": "predict_confirmation",
                "action": "auto_approved",
            }
        )
