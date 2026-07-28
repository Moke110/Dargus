"""Ingest workflow — hook-orchestrated evidence ingestion into D-Base.

Parses input sources, distributes content to DomainExperts for evidence
extraction, writes validated records into D-Base, and handles duplicate
review with an optional user confirmation gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from dargus.dbase.store import DuplicateReviewRequest
from dargus.dbase.validate import validate_evidence
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
            logger.exception("_run_ingest failed — returning empty report")
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
    4. **Input phase**: validate → dedup → embed → write via single-writer
    5. Collect DuplicateReviewRequest items
    6. Present duplicates for user confirmation
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

    # ---- Input phase: validate, dedup, embed, write via single-writer --------
    duplicate_review_requests: list[dict[str, Any]] = []
    records_written = 0
    records_skipped = 0

    store = task_spec.get("_dbase_store")
    if store is not None:
        extracted_instances = task_spec.get("_extracted_instances", [])
        records_written, records_skipped, duplicate_review_requests = _input_phase(
            store, extracted_instances
        )
        n_records = records_written
        n_errors = records_skipped
    # else: keep the stub counts (n_records from _extract_evidence stub)

    n_duplicates = len(duplicate_review_requests)

    # ---- Duplicate review -----------------------------------------------------
    duplicates = duplicate_review_requests
    if not duplicates and store is None:
        duplicates = _collect_duplicates(parsed_records, task_spec)
        n_duplicates = len(duplicates)

    # ---- Build FinalReport ----------------------------------------------------
    final_report: dict[str, Any] = {
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "source_path": source_path,
    }
    if isinstance(ctx.session, dict):
        ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks ----------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    # ---- User confirmation gate -----------------------------------------------
    if n_duplicates > 0:
        _user_confirmation_gate(ctx, task_spec, duplicates)
    elif store is not None:
        # Even with zero duplicates, append a confirmation entry (stub path)
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
    return []


# ---------------------------------------------------------------------------
# Input phase — validate, dedup, embed, write via single-writer
# ---------------------------------------------------------------------------


def _input_phase(
    store: Any,
    instances: list[dict[str, Any]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Persist extracted instances through the full single-writer lifecycle.

    1. ``validate_evidence()`` — invalid records (hard rejects) are logged and skipped
    2. ``write_record(…, dedup=True)`` — exact hash match returns ``False`` (skip);
       semantic dedup returns a ``DuplicateReviewRequest`` soft flag
    3. Embedding is generated on write so the embeddings sidecar stays current

    Args:
        store: A ``DBaseStore`` instance.
        instances: List of three-axis evidence dicts to persist.

    Returns:
        ``(records_written, records_skipped, duplicate_review_requests)``
    """
    n_written = 0
    n_skipped = 0
    duplicates: list[dict[str, Any]] = []

    for instance in instances:
        # 1. Validate — hard rejects are logged and skipped (not fatal)
        validation = validate_evidence(instance)
        if not validation.ok:
            logger.warning(
                "Validation failed for instance (skipped): %s",
                "; ".join(validation.hard_errors),
            )
            n_skipped += 1
            continue

        # 2. Write through single-writer: validate → evidence_id → dedup → embed → locked append
        try:
            result = store.write_record(instance, dedup=True)
        except Exception:
            logger.exception("write_record raised — skipping instance")
            n_skipped += 1
            continue

        if result is True:
            n_written += 1
        elif result is False:
            # Exact duplicate (evidence_id collision) — skip
            logger.info("Exact duplicate skipped for evidence_id=%s", instance.get("evidence_id"))
            n_skipped += 1
        elif isinstance(result, DuplicateReviewRequest):
            # Near-duplicate — soft flag
            logger.info(
                "Near-duplicate flagged: %s (similarity=%.3f)",
                result.candidate_evidence_id,
                result.similarity_score,
            )
            # The record was written (write_record returns DuplicateReviewRequest after writing)
            n_written += 1
            duplicates.append(
                {
                    "evidence_id": instance.get("evidence_id", ""),
                    "candidate_evidence_id": result.candidate_evidence_id,
                    "similarity_score": result.similarity_score,
                    "reason": "semantic_similarity",
                }
            )

    return n_written, n_skipped, duplicates


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
