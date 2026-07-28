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
    runtime: Any = None,
) -> dict[str, Any]:
    """Execute predict workflow with Hook enforcement.

    1. SESSION_START: SessionInitHook creates PredictSession
    2. Create D4Expert via AgentFactory from *runtime* (or stub)
    3. Main loop (max *max_rounds* / timeout *timeout_seconds*):
       - PERCEIVE_START: SkeletonContextHook injects state
       - Dispatch records to Domain Experts, collect ExpertReports
       - Process TaskDelegations across rounds until convergence
       - ROUND_END: SafetyNetHook checks convergence
    4. D4Expert.conclude() synthesizes FinalReport (DES +- DCS)
    5. SESSION_END: ReportValidationHook validates FinalReport
    6. ResultReportHook assembles PredictResult

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"predict"``),
            ``drug_ids``, ``disease_id``, optional ``endpoints``,
            ``max_rounds``, ``timeout_seconds``, ``require_confirmation``.
        runtime: Optional :class:`DargusRuntime`. When provided and carrying
            a ``dbase_store``, the real multi-round Expert loop runs;
            otherwise the stub path is used.

    Returns:
        PredictResult dict with keys: ``workflow``, ``status``,
        ``rounds_completed``, ``efficacy_score``, ``confidence_score``,
        ``drug_ids``, ``disease_id``, ``endpoints``, ``report``, ``session``.
    """
    task_spec.setdefault("workflow", "predict")
    max_rounds = int(task_spec.get("max_rounds", 5))
    timeout_seconds = float(task_spec.get("timeout_seconds", 300.0))

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

    # ---- Create initial context ------------------------------------------------
    ctx = HookContext(runtime=runtime, task_spec=task_spec)

    # ---- SESSION_START hooks --------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_START, ctx)

    # ---- Run the real round loop when a D-Base store is wired ---------------
    if _has_dbase_store(runtime):
        return _run_real_loop(ctx, runtime, hooks, max_rounds, timeout_seconds, task_spec)

    # ---- Backward-compat stub path (no D-Base store) -------------------------
    return _run_stub_loop(ctx, hooks, max_rounds, timeout_seconds, task_spec)


def _has_dbase_store(runtime: Any) -> bool:
    """True when *runtime* carries a usable DBaseStore."""
    try:
        from dargus.runtime.context import DargusRuntime

        if runtime is not None and isinstance(runtime, DargusRuntime):
            store = getattr(runtime, "dbase_store", None)
            return store is not None
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Real multi-round Expert loop (S4_T2)
# ---------------------------------------------------------------------------

# Domain keys ordered so that molecular runs first (the most fundamental
# level) and clinical runs last (the most applied).
_DOMAIN_ORDER = ["molecular", "biomedical", "bioinformatics", "clinical"]

_DOMAIN_TO_EXPERT_NAME: dict[str, str] = {
    "molecular": "MoleculeExpert",
    "biomedical": "BiomedExpert",
    "bioinformatics": "BioinfoExpert",
    "clinical": "ClinicExpert",
}


def _run_real_loop(
    ctx: HookContext,
    runtime: Any,
    hooks: HookRegistry,
    max_rounds: int,
    timeout_seconds: float,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Execute the real multi-round Expert dispatch/assess/delegate loop.

    1. Fetch relevant records from the D-Base store.
    2. Create D4Expert + Domain Experts via the AgentFactory.
    3. Round 0: each Expert gets ALL records; each processes its own
       biological levels and produces delegations for out-of-scope records.
    4. Rounds 1+: process ``TaskDelegation`` from the previous round,
       dispatching record subsets to the target Expert.
    5. Loop terminates when no new delegations or max_rounds is hit.
    6. D4Expert.conclude() synthesizes the FinalReport.
    """
    from dargus.runtime.factory import AgentFactory

    drug_ids = task_spec.get("drug_ids", [])
    disease_id = task_spec.get("disease_id", "unknown")
    endpoints = task_spec.get("endpoints", [])

    # ---- Fetch evidence records -------------------------------------------
    store = runtime.dbase_store
    records = store.read_records(disease_id=disease_id)

    # ---- Build Agents -----------------------------------------------------
    factory = AgentFactory(runtime)
    d4 = factory.d4_expert()
    experts: dict[str, Any] = {domain: factory.expert(domain) for domain in _DOMAIN_ORDER}

    # ---- Round loop -------------------------------------------------------
    round_num = 0
    # per_expert_report[expert_name] = list of ExpertReport per round
    per_expert_report: dict[str, list[Any]] = {}
    for domain in _DOMAIN_ORDER:
        expert = experts[domain]
        name = getattr(expert, "name", domain)
        per_expert_report[name] = []

    # pending_delegations_for_round[r] = list of TaskDelegation to process in round r
    pending_queue: list[list[Any]] = []

    # Initial queue: seed every domain with all records
    # (experts will scope-filter via can_handle)
    initial_delegations = _make_initial_delegations(records, _DOMAIN_ORDER)
    pending_queue.append(initial_delegations)

    # Cycle guard: track (target_expert, frozenset(record_ids)) already seen
    seen_delegations: set[tuple[str, frozenset[str]]] = set()

    while round_num < max_rounds:
        ctx.round = round_num

        # PERCEIVE_START
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)

        round_t0 = time.monotonic()

        # --- Get this round's delegations ----------------------------------
        round_delegations: list[Any] = []
        if round_num < len(pending_queue):
            round_delegations = pending_queue[round_num]

        # --- Dispatch: each delegation → target expert's assess() ---------
        new_delegations: list[Any] = []
        delegation_count = 0

        for delegation in round_delegations:
            target_domain = _delegation_domain(delegation)
            expert = experts.get(target_domain)
            if expert is None:
                logger.debug("No expert for domain %r — skipping delegation", target_domain)
                continue

            delegation_records = _delegation_records(delegation, records)

            # Build ExpertContext
            from dargus.experts.protocol import ExpertContext

            expert_ctx = ExpertContext(
                drug_ids=drug_ids,
                disease_id=disease_id,
                endpoints=endpoints,
                round=round_num,
                history=per_expert_report.get(target_domain, []),
            )

            # Run expert.assess()
            try:
                report = expert.assess(delegation_records, expert_ctx)
            except Exception as exc:
                logger.warning("Expert %s assessment failed: %s", target_domain, exc)
                continue

            # Collect the ExpertReport under the expert's class name
            expert_name = getattr(expert, "name", target_domain)
            per_expert_report.setdefault(expert_name, []).append(report)

            # Fire DOMAIN_REPORT_PRODUCED hook
            if hooks is not None:
                hooks.run(
                    HookPoint.DOMAIN_REPORT_PRODUCED,
                    HookContext(
                        runtime=ctx.runtime,
                        task_spec=ctx.task_spec,
                        session=ctx.session,
                        round=round_num,
                        extra={"domain_report": _report_to_dict(report)},
                    ),
                )

            # Queue new delegations for the next round (cycle-guarded)
            for d in report.delegations:
                delegation_key = _delegation_key(d)
                if delegation_key not in seen_delegations:
                    seen_delegations.add(delegation_key)
                    new_delegations.append(d)
                    delegation_count += 1

        # --- Store new delegations for next round --------------------------
        if new_delegations:
            pending_queue.append(new_delegations)

        # --- D4Expert optionally synthesizes or provides guidance ----------
        ctx.extra["round_elapsed_ms"] = (time.monotonic() - round_t0) * 1000

        # Store round summary in session
        if isinstance(ctx.session, dict):
            rounds = ctx.session.setdefault("rounds", [])
            rounds.append(
                {
                    "round": round_num,
                    "report_summary": f"Round {round_num}: "
                    f"{sum(1 for rl in per_expert_report.values() for _ in rl)} "
                    f"expert reports collected, "
                    f"{delegation_count} new delegations",
                }
            )

        # ROUND_END safety check
        ctx = hooks.run(HookPoint.ROUND_END, ctx)
        round_num += 1

        # Stop if no new delegations (converged) or safety net triggered
        if ctx.extra.get("force_converge"):
            logger.info("Safety net triggered — forcing convergence at round %d", round_num - 1)
            break
        if not new_delegations and round_num > 0:
            logger.info("No new delegations — converged at round %d", round_num - 1)
            break

    # ---- Synthesize FinalReport ------------------------------------------
    endpoint = endpoints[0] if endpoints else "efficacy"
    final = d4.conclude(
        drug_id=drug_ids[0] if drug_ids else "unknown",
        disease_id=disease_id,
        endpoint=endpoint,
        all_reports=per_expert_report,
    )

    # ---- Store FinalReport in context ------------------------------------
    final_report_dict = _final_report_to_dict(final)
    ctx.session["FinalReport"] = final_report_dict  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report_dict

    # ---- Register validation + result hooks (after the fact so they ---------
    # ---- have access to the runtime's D-Base stored in the runtime) --------
    validation_dbase = store.dbase if store is not None else _hook_dbase()
    hooks.register(HookPoint.SESSION_END, ReportValidationHook(dbase=validation_dbase))
    hooks.register(HookPoint.SESSION_END, ResultReportHook())

    # ---- SESSION_END hooks (validation + report assembly) ----------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate ------------------------------------------
    _user_confirmation_gate(ctx, task_spec)

    # ---- Return result ---------------------------------------------------
    return {
        "workflow": "predict",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "rounds_completed": round_num,
        "efficacy_score": final.efficacy_score,
        "confidence_score": final.confidence_score,
        "drug_ids": [final.drug_id],
        "disease_id": final.disease_id,
        "endpoints": [final.endpoint] if final.endpoint else [],
        "report": final_report_dict,
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Backward-compat stub loop (no D-Base store)
# ---------------------------------------------------------------------------


def _run_stub_loop(
    ctx: HookContext,
    hooks: HookRegistry,
    max_rounds: int,
    timeout_seconds: float,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Stub predict loop: runs the stub D4Expert for backward compatibility.

    This path is used when no DargusRuntime is provided with a dbase_store.
    It retains the original S4_T1 stub behavior.
    """
    # ---- Build stub D4Expert -----------------------------------------------
    d4 = _StubD4Expert(hooks)

    # ---- Main round loop --------------------------------------------------
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
            logger.info(
                "Safety net triggered — forcing convergence at round %d",
                round_num - 1,
            )
            break

    # ---- Store FinalReport in context -------------------------------------
    final_report = _build_final_report(report, task_spec)
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- Register validation + result hooks (stub path) ---------------------
    hooks_stub_dbase = _hook_dbase()
    hooks.register(HookPoint.SESSION_END, ReportValidationHook(dbase=hooks_stub_dbase))
    hooks.register(HookPoint.SESSION_END, ResultReportHook())

    # ---- SESSION_END hooks (validation + report assembly) -----------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate (stub: auto-approve) ----------------------
    _user_confirmation_gate(ctx, task_spec)

    # ---- Return result ----------------------------------------------------
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


def _make_initial_delegations(
    records: list[dict],
    domains: list[str],
) -> list[Any]:
    """Create fake 'delegations' for round 0 that seed every domain with all records.

    In the initial round every Expert sees all records and filters
    via ``can_handle()``.  This keeps the loop uniform: each round is a
    list of (target_domain, subset_of_records) pairs.
    """
    from dargus.experts.protocol import TaskDelegation

    delegations: list[Any] = []
    for domain in domains:
        # Use the domain name as the reason text (it will be decoded by
        # _delegation_domain / _delegation_records below)
        expert_name = _DOMAIN_TO_EXPERT_NAME.get(domain, domain.capitalize() + "Expert")
        delegations.append(
            TaskDelegation(
                target_expert=expert_name,
                record_ids=[r.get("evidence_id", "") for r in records],
                reason=f"Initial dispatch to {domain}",
                priority="high",
            )
        )
    return delegations


def _delegation_domain(delegation: Any) -> str:
    """Map a TaskDelegation's target_expert name back to a domain key."""
    from dargus.experts.protocol import TaskDelegation

    if isinstance(delegation, TaskDelegation):
        target = delegation.target_expert.lower()
    elif isinstance(delegation, dict):
        target = (delegation.get("target_expert") or "").lower()
    else:
        return ""
    # Map expert names → domain keys
    mapping = {
        "moleculeexpert": "molecular",
        "biomedexpert": "biomedical",
        "bioinfoexpert": "bioinformatics",
        "clinicexpert": "clinical",
    }
    return mapping.get(target, "")


def _delegation_records(delegation: Any, all_records: list[dict]) -> list[dict]:
    """Return the records referenced by a TaskDelegation, looked up from *all_records*."""
    from dargus.experts.protocol import TaskDelegation

    if isinstance(delegation, TaskDelegation):
        rids = set(delegation.record_ids)
    elif isinstance(delegation, dict):
        rids = set(delegation.get("record_ids", []))
    else:
        return list(all_records)  # unknown form — pass everything
    if not rids:
        return list(all_records)
    return [r for r in all_records if r.get("evidence_id") in rids]


def _delegation_key(delegation: Any) -> tuple[str, frozenset[str]]:
    """Return a cycle-guard key for a TaskDelegation."""
    from dargus.experts.protocol import TaskDelegation

    if isinstance(delegation, TaskDelegation):
        return (
            delegation.target_expert.lower(),
            frozenset(delegation.record_ids),
        )
    if isinstance(delegation, dict):
        return (
            (delegation.get("target_expert") or "").lower(),
            frozenset(delegation.get("record_ids", [])),
        )
    return ("", frozenset())


def _report_to_dict(report: Any) -> dict[str, Any]:
    """Convert an ExpertReport (dataclass or dict) to a plain dict."""
    if isinstance(report, dict):
        return report
    return {
        "expert": getattr(report, "expert", ""),
        "round": getattr(report, "round", 0),
        "findings": [
            {
                "record_ids": f.record_ids,
                "biological_level": f.biological_level,
                "quality_score": f.quality_score,
            }
            for f in getattr(report, "findings", [])
        ],
        "confidence": {
            "low": (
                getattr(report, "confidence").low if getattr(report, "confidence", None) else 0.0
            ),
            "high": (
                getattr(report, "confidence").high if getattr(report, "confidence", None) else 1.0
            ),
        },
        "delegations": [
            {
                "target_expert": d.target_expert,
                "record_ids": d.record_ids,
                "reason": d.reason,
            }
            for d in getattr(report, "delegations", [])
        ],
        "data_gaps": getattr(report, "data_gaps", []),
        "bias_notes": getattr(report, "bias_notes", []),
    }


def _final_report_to_dict(final: Any) -> dict[str, Any]:
    """Convert a FinalReport dataclass to a plain dict for the session/return."""
    return {
        "drug_ids": [getattr(final, "drug_id", "")],
        "disease_id": getattr(final, "disease_id", ""),
        "endpoints": [getattr(final, "endpoint", "")] if getattr(final, "endpoint", None) else [],
        "efficacy_score": getattr(final, "efficacy_score", None),
        "confidence_score": getattr(final, "confidence_score", None),
        "confidence_level": getattr(final, "confidence_level", "low"),
        "overall_conclusion": getattr(final, "expert_consensus", ""),
        "supporting_records": getattr(final, "supporting_records", []),
        "reasoning_mode": getattr(final, "reasoning_mode", "Iris-expert"),
        "contradictions": getattr(final, "contradictions", []),
        "data_gaps": getattr(final, "data_gaps", []),
    }


# ---------------------------------------------------------------------------
# Stub loop internals (kept for backward compat)
# ---------------------------------------------------------------------------


class _StubD4Expert:
    """Minimal D4Expert stub that provides delegate_to_expert + synthesize."""

    def __init__(self, hooks: HookRegistry) -> None:
        self._hooks = hooks

    def delegate_to_expert(self, domain: str, records: list[dict], question: str) -> dict[str, Any]:
        return {
            "domain": domain,
            "conclusion": f"Stub assessment for {domain}: {question[:80]}",
            "confidence": {"low": 0.5, "high": 0.8},
            "supporting_evidence": [],
        }

    def synthesize(self, expert_reports: list[dict[str, Any]]) -> dict[str, Any]:
        confidences: list[float] = []
        for r in expert_reports:
            c = r.get("confidence", {})
            if isinstance(c, dict):
                confidences.append(c.get("low", 0.0))
                confidences.append(c.get("high", 1.0))
        avg = sum(confidences) / len(confidences) if confidences else 0.5
        return {
            "overall_conclusion": f"Synthesized {len(expert_reports)} expert reports",
            "confidence": "moderate" if avg > 0.3 else "low",
            "expert_reports": expert_reports,
            "conflicts": [],
        }


def _execute_predict_round(
    ctx: HookContext,
    d4: Any,
    round_num: int,
    hooks: HookRegistry | None = None,
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
