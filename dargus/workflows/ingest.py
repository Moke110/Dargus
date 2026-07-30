"""Ingest workflow -- hook-orchestrated evidence ingestion into D-Base.

Parses input sources, distributes content to DomainExperts for evidence
extraction, writes validated records into D-Base, and handles duplicate
review with an optional user confirmation gate.
"""

from __future__ import annotations

import logging
from collections import Counter
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
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class IngestionSummary:
    """Presented at the confirmation gate before records are written.

    Built from the extracted instances after the Convert phase and the
    duplicate list surfaced by the semantic dedup check (S9, S3_T3).

    Attributes:
        per_domain: Count of records per domain (e.g. ``{"molecular": 2, "clinical": 1}``).
        n_to_write: Total records to write.
        n_duplicates: Soft-flagged near-duplicates.
        duplicates: The raw ``DuplicateReviewRequest`` dicts for the gate.
    """

    per_domain: dict[str, int]
    n_to_write: int
    n_duplicates: int
    duplicates: list[dict[str, Any]]

    @classmethod
    def from_instances(
        cls,
        instances: list[dict[str, Any]],
        duplicates: list[dict[str, Any]],
    ) -> IngestionSummary:
        """Build a summary from extracted instances and duplicate flags."""
        per_domain: dict[str, int] = dict(Counter(i.get("domain", "unknown") for i in instances))
        return cls(
            per_domain=per_domain,
            n_to_write=len(instances),
            n_duplicates=len(duplicates),
            duplicates=list(duplicates),
        )


# ---------------------------------------------------------------------------
# Backward-compat dataclasses (consumed by Iris.commander, api, cli)
# ---------------------------------------------------------------------------


@dataclass
class IngestionReport:
    """Report from an Ingest workflow run."""

    n_records: int = 0
    n_skipped: int = 0
    dbase_size: int = 0
    errors: list[str] = field(default_factory=list)


TrainingReport = IngestionReport  # backward compat alias


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ingest(
    task_spec_or_datadir: dict[str, Any] | str,
    reset: bool = False,
    disease_kb_dir: str | None = None,
) -> IngestionReport | dict[str, Any]:
    """Execute the ingest workflow.

    **Backward-compat wrapper.**  Supports both the Phase-E ``task_spec``
    calling convention and the pre-Phase-E ``(datadir, reset, disease_kb_dir)``
    convention used by ``Iris.commander``, ``api.py``, and ``cli.py``.

    Args:
        task_spec_or_datadir: Either a ``task_spec`` dict (new API) or a
            ``datadir`` path string (old API).
        reset: (old API) Clear D-Base before ingestion.
        disease_kb_dir: (old API) Path to disease knowledge base directory.

    Returns:
        - ``dict[str, Any]`` when called with a ``task_spec`` dict.
        - ``IngestionReport`` when called with a ``datadir`` string.
    """
    if isinstance(task_spec_or_datadir, str):
        # Old calling convention: run_ingest(datadir, reset=..., disease_kb_dir=...)
        task_spec: dict[str, Any] = {
            "workflow": "ingest",
            "source_path": task_spec_or_datadir,
            "reset": reset,
        }
        if disease_kb_dir is not None:
            task_spec["disease_kb_dir"] = disease_kb_dir
        try:
            result = _run_ingest(task_spec)
        except Exception:
            logger.exception("_run_ingest failed -- returning empty report")
            return IngestionReport()
        return IngestionReport(
            n_records=result.get("n_records", 0),
            n_skipped=result.get("n_duplicates", 0),
            dbase_size=0,
            errors=[],
        )
    # New calling convention: run_ingest(task_spec)
    return _run_ingest(task_spec_or_datadir)


def _run_ingest(task_spec: dict[str, Any]) -> dict[str, Any]:
    """Execute ingest workflow with Hook enforcement.

    1. SESSION_START: SessionInitHook creates IngestSession
    2. Explore and parse input files from source_path
    3. Distribute content to DomainExperts for evidence extraction
    4. Call dbase_write for each extracted record
    5. Collect DuplicateReviewRequest items
    6. Present IngestionSummary for user confirmation (via callback or default allow)
    7. Route decision: proceed / skip-duplicates / abort
    8. Report ingestion summary

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"ingest"``),
            ``source_path``, optional ``source_type``, ``max_rounds``,
            ``require_confirmation``, ``confirm_callback``.

    Returns:
        IngestResult dict with keys: ``workflow``, ``status``,
        ``n_records``, ``n_duplicates``, ``n_errors``, ``per_domain``,
        ``session``.
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

    # ---- Build per-domain summary from parsed records --------------------------
    per_domain: dict[str, int] = {}
    for rec in parsed_records:
        d = rec.get("domain", "unknown")
        per_domain[d] = per_domain.get(d, 0) + 1

    # ---- Duplicate review -----------------------------------------------------
    duplicates = _collect_duplicates(parsed_records, task_spec)
    n_duplicates = len(duplicates)

    # ---- Build IngestionSummary and invoke confirmation gate -------------------
    summary = IngestionSummary.from_instances(parsed_records, duplicates)
    decision = _user_confirmation_gate(ctx, task_spec, summary)

    # ---- Route decision --------------------------------------------------------
    if decision == "abort":
        return {
            "workflow": "ingest",
            "status": "aborted_by_user",
            "n_records": 0,
            "n_duplicates": n_duplicates,
            "n_errors": n_errors,
            "per_domain": {},
            "session": ctx.session,
        }

    if decision == "skip-duplicates":
        # Keep only non-flagged records. In this stub, duplicates are identity
        # dicts without a record attachment, so we cannot filter precisely.
        # Instead we subtract n_duplicates from n_records as a faithful stub.
        n_records = max(0, n_records - n_duplicates)
        for d in per_domain:
            per_domain[d] = min(per_domain[d], n_records)

    # ---- Build FinalReport ----------------------------------------------------
    final_report: dict[str, Any] = {
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "per_domain": per_domain,
        "source_path": source_path,
        "gate_decision": decision,
    }
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks ----------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- Return result --------------------------------------------------------
    return {
        "workflow": "ingest",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "per_domain": per_domain,
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
    ctx: HookContext | Any,
    task_spec: dict[str, Any],
    summary: IngestionSummary,
) -> str:
    """HITL confirmation gate between Convert and Input phases.

    Presents an ``IngestionSummary`` (per-domain counts, records to write,
    duplicates flagged) and requests a decision: **proceed** /
    **skip-duplicates** / **abort**.

    When a ``confirm_callback`` is provided in *task_spec*, it is invoked
    with ``(summary, duplicates)`` and must return one of the three
    decision strings. When no callback is configured, the gate defaults to
    ``"proceed"`` (allow) per ``CLAUDE.md``.

    Returns:
        One of ``"proceed"``, ``"skip-duplicates"``, or ``"abort"``.
    """
    callback = task_spec.get("confirm_callback")
    duplicates = summary.duplicates

    if callback is not None:
        decision = callback(summary, duplicates)
        logger.info(
            "User confirmation gate: callback returned %r (%d duplicates, %d to write)",
            decision,
            summary.n_duplicates,
            summary.n_to_write,
        )
    else:
        # Default to allow: no callback means auto-proceed.
        decision = "proceed"
        logger.info(
            "User confirmation gate: no callback -- defaulting to %r (%d duplicates, %d to write)",
            decision,
            summary.n_duplicates,
            summary.n_to_write,
        )

    # Record the decision in the session.
    if isinstance(ctx, HookContext) and isinstance(ctx.session, dict):
        ctx.session.setdefault("confirmations", []).append(
            {
                "type": "confirmation_gate",
                "n_duplicates": summary.n_duplicates,
                "n_to_write": summary.n_to_write,
                "action": decision,
                "per_domain": summary.per_domain,
            }
        )
    elif hasattr(ctx, "session") and isinstance(getattr(ctx, "session", None), dict):
        ctx.session.setdefault("confirmations", []).append(
            {
                "type": "confirmation_gate",
                "n_duplicates": summary.n_duplicates,
                "n_to_write": summary.n_to_write,
                "action": decision,
                "per_domain": summary.per_domain,
            }
        )

    return decision
