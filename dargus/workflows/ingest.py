"""Ingest workflow — hook-orchestrated evidence ingestion into D-Base.

Parses input sources, distributes content to DomainExperts for evidence
extraction, writes validated records into D-Base, and handles duplicate
review with an optional user confirmation gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestionReport:
    """Report from an Ingest workflow run."""

    n_records: int = 0
    n_skipped: int = 0
    dbase_size: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ingest(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Execute the ingest workflow.

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"ingest"``),
            ``source_path``, optional ``source_type``, ``max_rounds``,
            ``require_confirmation``.

    Returns:
        IngestResult dict with keys: ``workflow``, ``status``,
        ``n_records``, ``n_duplicates``, ``n_errors``, ``session``.
    """
    return _run_ingest(task_spec)


def _run_ingest(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Execute ingest workflow with Hook enforcement.

    1. SESSION_START: SessionInitHook creates IngestSession
    2. Explore and parse input files from source_path
    3. Distribute content to DomainExperts for evidence extraction
    4. Call dbase_write for each extracted record
    5. Collect DuplicateReviewRequest items
    6. Present duplicates for user confirmation (stub: auto-approve)
    7. Report ingestion summary

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"ingest"``),
            ``source_path``, optional ``source_type``, ``max_rounds``,
            ``require_confirmation``.

    Returns:
        IngestResult dict with keys: ``workflow``, ``status``,
        ``n_records``, ``n_duplicates``, ``n_errors``, ``session``.
    """
    task_spec.setdefault("workflow", "ingest")
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

    # ---- Parse source ----------------------------------------------------------
    source_path = task_spec.get("source_path", "")
    logger.info("Ingesting from source: %s", source_path)

    # Stub parsing: simulate file exploration and content extraction
    parsed_records = _parse_source(source_path)
    n_records = 0
    n_duplicates = 0
    n_errors = 0

    # ---- Main round loop: distribute to DomainExperts -------------------------
    round_num = 0
    domain_records = _partition_by_domain(parsed_records)

    while round_num < max_rounds:
        ctx.round = round_num
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)

        # In each round, process one domain batch
        if round_num < len(domain_records):
            domain, batch = domain_records[round_num]
            extracted, errors = _extract_evidence(domain, batch)
            n_records += extracted
            n_errors += errors

            if isinstance(ctx.session, dict):
                rounds = ctx.session.setdefault("rounds", [])
                rounds.append(
                    {
                        "round": round_num,
                        "domain": domain,
                        "extracted": extracted,
                        "errors": errors,
                    }
                )

        ctx = hooks.run(HookPoint.ROUND_END, ctx)
        if ctx.extra.get("force_converge"):
            break
        round_num += 1

    # ---- Duplicate review -----------------------------------------------------
    duplicates = _collect_duplicates(parsed_records, task_spec)
    n_duplicates = len(duplicates)

    # ---- Build FinalReport ----------------------------------------------------
    final_report: dict[str, Any] = {
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "source_path": source_path,
    }
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks ----------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate -----------------------------------------------
    if n_duplicates > 0:
        _user_confirmation_gate(ctx, task_spec, duplicates)

    # ---- Return result --------------------------------------------------------
    return {
        "workflow": "ingest",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_source(source_path: str) -> list[dict[str, Any]]:
    """Parse input source into a list of raw evidence record dicts.

    Currently a stub that returns synthetic records for testing.
    """
    if not source_path:
        return []

    # Stub: return a set of synthetic records
    return (
        [
            {"id": f"rec-{i:03d}", "source": source_path, "domain": "molecular", "data": {}}
            for i in range(5)
        ]
        + [
            {"id": f"rec-{i:03d}", "source": source_path, "domain": "biomedical", "data": {}}
            for i in range(5, 10)
        ]
        + [
            {"id": f"rec-{i:03d}", "source": source_path, "domain": "clinical", "data": {}}
            for i in range(10, 15)
        ]
    )


def _partition_by_domain(
    records: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group records by domain key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        domain = rec.get("domain", "unknown")
        groups.setdefault(domain, []).append(rec)
    return list(groups.items())


def _extract_evidence(domain: str, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Extract evidence from records using DomainExpert (stub).

    Returns (n_extracted, n_errors).
    """
    # Stub: every record extracts successfully, no errors
    logger.info("Domain %s: extracting evidence from %d records", domain, len(records))
    return len(records), 0


def _collect_duplicates(
    records: list[dict[str, Any]], task_spec: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Identify duplicate records.

    Returns injected duplicates from ``_duplicate_records`` in *task_spec*
    when provided (for testing).  Otherwise returns an empty list (stub).
    """
    if task_spec is not None and task_spec.get("_duplicate_records"):
        return list(task_spec["_duplicate_records"])
    # In a real implementation this would query D-Base for existing records
    # with matching fingerprints and return DuplicateReviewRequest items.
    return []


def _user_confirmation_gate(
    ctx: HookContext, task_spec: dict[str, Any], duplicates: list[dict[str, Any]]
) -> None:
    """Stub user confirmation for duplicate review.

    Set ``require_confirmation: true`` in *task_spec* to enable the log
    message. In the future this will prompt for actual user input.
    """
    if task_spec.get("require_confirmation"):
        logger.info(
            "User confirmation required for %d duplicates — auto-approved (stub)",
            len(duplicates),
        )
    if isinstance(ctx.session, dict):
        ctx.session.setdefault("confirmations", []).append(
            {
                "type": "duplicate_review",
                "n_duplicates": len(duplicates),
                "action": "auto_approved",
            }
        )
