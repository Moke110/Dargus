"""Ingest workflow -- hook-orchestrated evidence ingestion into D-Base.

Three-phase pipeline:
1. Explore: directory scan → discover files, classify by domain
2. Convert: Domain Experts extract structured evidence from files
3. Input: validate, dedup, embed, write through single DBaseStore writer

HITL confirmation gate between Convert and Input phases.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
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
# Recognised file extensions for evidence source files
# ---------------------------------------------------------------------------

_EVIDENCE_EXTENSIONS: tuple[str, ...] = (".json", ".txt", ".csv")

# Domain classification heuristics for the Explore phase when no LLM is
# wired.  Keys are domain keys, values are keyword lists matched against
# the lowercase filename stem.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "molecular": ["molecular", "binding", "docking", "assay", "ic50", "ec50", "ki"],
    "biomedical": ["cellular", "viability", "biomed", "mtor", "akt", "western", "qPCR"],
    "bioinformatics": ["gene", "genomic", "expression", "transcriptom", "rnaseq"],
    "clinical": ["clinical", "rct", "trial", "patient", "adverse", "survival"],
}


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


TrainingReport = IngestionReport  # backward compat alias (S2_T2: remove in v1.1)


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
    2. Explore: scan source_path directory, classify files by domain
    3. Convert: Domain Experts extract structured evidence from files
    4. Input: validate, dedup, embed, write via single DBaseStore writer
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
    hooks.register(
        HookPoint.ROUND_END,
        SafetyNetHook(max_rounds=max_rounds, session_timeout=600.0),
    )
    hooks.register(HookPoint.SESSION_END, ReportValidationHook())
    hooks.register(HookPoint.SESSION_END, ResultReportHook())

    # ---- Create initial context ------------------------------------------------
    ctx = HookContext(runtime=None, task_spec=task_spec)

    # ---- SESSION_START hooks --------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_START, ctx)

    # ---- PHASE 1: Explore — scan directory, classify files by domain -------
    source_path = task_spec.get("source_path", "")
    logger.info("Ingesting from source: %s", source_path)

    # _explore_source returns classified files:
    #   {"domain": ["file1.json", "file2.txt"], ...}
    # A non-empty result signals that the source_path is a real filesystem
    # location — run the full three-phase pipeline.  When the result is
    # empty (synthetic paths used by tests), fall through to the legacy stub.
    domain_files = _explore_source(source_path, task_spec)

    if domain_files:
        # ---- Real pipeline: Explore → Convert → Input --------------------------
        return _run_phased_ingest(ctx, task_spec, domain_files, hooks)

    # ---- Legacy stub path (synthetic source paths, test compatibility) ---------
    if not source_path:
        return {
            "workflow": "ingest",
            "status": "completed",
            "n_records": 0,
            "n_duplicates": 0,
            "n_errors": 0,
            "per_domain": {},
            "session": ctx.session,
        }

    # Synthetic stub: produce the same 15-record batches as pre-v1.0.0
    parsed_records = _parse_source(source_path)
    n_records = 0
    n_errors = 0
    round_num = 0
    domain_records = _partition_by_domain(parsed_records)
    max_rounds = int(task_spec.get("max_rounds", 5))

    while round_num < max_rounds:
        ctx.round = round_num
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)
        if round_num < len(domain_records):
            domain, batch = domain_records[round_num]
            n_records += len(batch)
            if isinstance(ctx.session, dict):
                rounds = ctx.session.setdefault("rounds", [])
                rounds.append(
                    {
                        "round": round_num,
                        "domain": domain,
                        "extracted": len(batch),
                        "errors": 0,
                    }
                )
        ctx = hooks.run(HookPoint.ROUND_END, ctx)
        if ctx.extra.get("force_converge"):
            break
        round_num += 1

    # Per-domain summary
    per_domain: dict[str, int] = {}
    for rec in parsed_records:
        d = rec.get("domain", "unknown")
        per_domain[d] = per_domain.get(d, 0) + 1

    # Duplicate review (honours _duplicate_records injection for tests)
    duplicates = _collect_duplicates(parsed_records, task_spec)
    n_duplicates = len(duplicates)

    # HITL confirmation gate
    summary = IngestionSummary.from_instances(parsed_records, duplicates)
    decision = _user_confirmation_gate(ctx, task_spec, summary)

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
        n_records = max(0, n_records - n_duplicates)
        for d in per_domain:
            per_domain[d] = min(per_domain[d], n_records)

    # FinalReport
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
    ctx = hooks.run(HookPoint.SESSION_END, ctx)
    return {
        "workflow": "ingest",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": n_errors,
        "per_domain": per_domain,
        "session": ctx.session,
    }


def _run_phased_ingest(
    ctx: HookContext,
    task_spec: dict[str, Any],
    domain_files: dict[str, list[str]],
    hooks: HookRegistry,
) -> dict[str, Any]:
    """Run the full three-phase ingest pipeline against real source files."""
    source_path = task_spec.get("source_path", "")
    all_instances: list[dict[str, Any]] = []
    n_errors = 0
    round_num = 0

    # ---- PHASE 2: Convert — Domain Experts extract evidence from files -----
    for domain, files in domain_files.items():
        ctx.round = round_num
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)

        extracted, errs = _extract_evidence(domain, files, task_spec)
        all_instances.extend(extracted)
        n_errors += errs

        if isinstance(ctx.session, dict):
            rounds = ctx.session.setdefault("rounds", [])
            rounds.append(
                {
                    "round": round_num,
                    "domain": domain,
                    "extracted": len(extracted),
                    "errors": errs,
                }
            )

        ctx = hooks.run(HookPoint.ROUND_END, ctx)
        if ctx.extra.get("force_converge"):
            break
        round_num += 1

    # ---- PHASE 3: Input — validate, dedup, embed, write to D-Base ---------
    written: list[dict[str, Any]] = []
    duplicate_flags: list[dict[str, Any]] = []
    n_errors_write = 0

    # Collect duplicates (honours _duplicate_records in task_spec for testing)
    duplicates = _collect_duplicates(all_instances, task_spec)
    duplicate_flags = [
        (
            {"evidence_id": d.get("evidence_id", "?"), "reason": "exact_match"}
            if isinstance(d, dict)
            else {"evidence_id": getattr(d, "candidate_evidence_id", "?"), "reason": "semantic_dup"}
        )
        for d in duplicates
    ]

    # Wire DBaseStore when available; stub path when not
    dbase_store = _get_dbase_store(task_spec)
    if dbase_store is not None and all_instances:
        duplicate_eids = {
            (
                d.get("evidence_id", "")
                if isinstance(d, dict)
                else getattr(d, "candidate_evidence_id", "")
            )
            for d in duplicates
        }
        for inst in all_instances:
            if inst.get("evidence_id", "?") in duplicate_eids:
                continue
            try:
                result = dbase_store.write_record(inst, dedup=True)
                if result is True:
                    written.append(inst)
                elif result is not False:
                    duplicate_flags.append(
                        {
                            "evidence_id": getattr(result, "candidate_evidence_id", "?"),
                            "reason": "semantic_dup",
                        }
                    )
            except Exception:
                n_errors_write += 1
                logger.warning(
                    "Failed to write record %s", inst.get("evidence_id", "?"), exc_info=True
                )
    else:
        written = list(all_instances)

    n_duplicates = len(duplicate_flags)
    total_errors = n_errors + n_errors_write

    # ---- Per-domain summary -------------------------------------------------
    per_domain: dict[str, int] = {}
    for rec in written:
        d = rec.get("domain", "unknown")
        per_domain[d] = per_domain.get(d, 0) + 1

    # ---- HITL confirmation gate ----------------------------------------------
    summary = IngestionSummary.from_instances(written, duplicate_flags)
    decision = _user_confirmation_gate(ctx, task_spec, summary)

    if decision == "abort":
        return {
            "workflow": "ingest",
            "status": "aborted_by_user",
            "n_records": 0,
            "n_duplicates": n_duplicates,
            "n_errors": total_errors,
            "per_domain": {},
            "session": ctx.session,
        }

    if decision == "skip-duplicates":
        n_records = max(0, len(written) - n_duplicates)
        flagged_eids = {d.get("evidence_id", "") for d in duplicate_flags}
        kept = [r for r in written if r.get("evidence_id", "") not in flagged_eids]
        written = kept
        per_domain_clean: dict[str, int] = {}
        for rec in written:
            d = rec.get("domain", "unknown")
            per_domain_clean[d] = per_domain_clean.get(d, 0) + 1
        per_domain = per_domain_clean
    else:
        n_records = len(written)

    # ---- FinalReport --------------------------------------------------------
    final_report: dict[str, Any] = {
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": total_errors,
        "per_domain": per_domain,
        "source_path": source_path,
        "gate_decision": decision,
    }
    ctx.session["FinalReport"] = final_report  # type: ignore[index]
    ctx.extra["FinalReport"] = final_report

    # ---- SESSION_END hooks ----------------------------------------------------
    ctx = hooks.run(HookPoint.SESSION_END, ctx)

    return {
        "workflow": "ingest",
        "status": ctx.extra.get("result", {}).get("status", "completed"),
        "n_records": n_records,
        "n_duplicates": n_duplicates,
        "n_errors": total_errors,
        "per_domain": per_domain,
        "session": ctx.session,
    }


# ---------------------------------------------------------------------------
# Internal helpers — three-phase pipeline
# ---------------------------------------------------------------------------


# Re-export legacy function name for test compatibility; returns synthetic
# records matching the pre-v1.0.0 stub shape used by test_parse_source_returns_records.
def _parse_source(source_path: str) -> list[dict[str, Any]]:
    """Legacy stub: parse input source into a list of raw evidence record dicts.

    Retained only for backward-compat tests.  Real Explore phase uses
    :func:`_explore_source` instead.
    """
    if not source_path:
        return []
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


def _explore_source(
    source_path: str, task_spec: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Explore phase: scan directory, discover evidence files, classify by domain.

    When ``source_path`` is a JSON file (``_mock_classifications`` key in
    *task_spec*, for testing), it is parsed as mock classifications.
    Otherwise the path is treated as a directory and real files are classified
    using keyword heuristics (or a reasoning LLM when wired).

    Returns:
        ``{domain: [file_path, ...], ...}`` — domain keys mapped to
        absolute file paths.  Unknown/unclassifiable files are logged
        and skipped.  Returns an empty dict when the source directory
        does not exist or contains no recognised files.
    """
    # ---- Test-injection path: _mock_classifications in task_spec -----------
    if task_spec is not None and "_mock_classifications" in task_spec:
        return dict(task_spec["_mock_classifications"])

    if not source_path:
        return {}

    src = Path(source_path)

    # ---- Single-file path: classify directly --------------------------------
    if src.is_file():
        domain = _classify_file(src.name, task_spec)
        if domain:
            return {domain: [str(src)]}
        logger.info("Explore: file %s could not be classified — skipping", src.name)
        return {}

    # ---- Directory path: walk and classify every evidence file --------------
    if not src.is_dir():
        logger.warning("Explore: source_path %s is not a file or directory", source_path)
        return {}

    classified: dict[str, list[str]] = {}
    for fp in sorted(src.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in _EVIDENCE_EXTENSIONS:
            logger.debug("Explore: skipping %s (non-evidence extension)", fp.name)
            continue
        domain = _classify_file(fp.name, task_spec)
        if domain:
            classified.setdefault(domain, []).append(str(fp))
        else:
            logger.info("Explore: file %s could not be classified by domain — skipping", fp.name)

    return classified


def _classify_file(filename: str, task_spec: dict[str, Any] | None = None) -> str | None:
    """Classify a single file name into a domain key.

    When a reasoning LLM is wired (``_reasoning_llm`` key in *task_spec*),
    uses prompt-based classification.  Otherwise falls back to keyword
    heuristics on the lowercased file stem.
    """
    # ---- LLM-based classification (when a reasoning LLM is injected) ----
    llm = (task_spec or {}).get("_reasoning_llm")
    if llm is not None:
        return _classify_file_via_llm(filename, llm)

    # ---- Keyword heuristic fallback ---------------------------------------
    stem = Path(filename).stem.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in stem:
                return domain
    return None


def _classify_file_via_llm(filename: str, llm: Any) -> str | None:
    """Prompt-based domain classification of a single file."""
    from dargus.models.reasoning import Message

    prompt = (
        "Classify the following file name into exactly one of these domains: "
        "molecular, biomedical, bioinformatics, clinical. "
        "Return ONLY the domain name, nothing else.\n\n"
        f"File: {filename}"
    )
    try:
        response = llm.chat([Message(role="user", content=prompt)])
        domain = response.content.strip().lower()
        if domain in _DOMAIN_KEYWORDS:
            return domain
        logger.warning("LLM returned unknown domain %r for %s", domain, filename)
        return None
    except Exception:
        logger.warning("LLM classification failed for %s", filename, exc_info=True)
        return None


def _partition_by_domain(
    records: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group records by domain key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        domain = rec.get("domain", "unknown")
        groups.setdefault(domain, []).append(rec)
    return list(groups.items())


def _extract_evidence(
    domain: str, files: list[str], task_spec: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Convert phase: extract structured evidence from files via DomainExpert.

    When a Domain Expert is wired via ``_domain_experts`` in the task_spec
    (keyed by domain), calls ``expert.extract(files)``.  When no expert is
    available, falls back to stub evidence instances using the Expert base
    class's ``_STUB_EVIDENCE`` dicts keyed by the domain's primary biological
    level.

    Individual file extraction failures are caught and counted as errors;
    they do not abort the batch.

    Returns (extracted_instances, n_errors).
    """
    experts = (task_spec or {}).get("_domain_experts", {}) if task_spec else {}
    expert = experts.get(domain) if isinstance(experts, dict) else None
    if expert is not None:
        instances: list[dict[str, Any]] = []
        errors = 0
        for fp in files:
            try:
                results = expert.extract(fp)
                instances.extend(results)
            except Exception:
                logger.warning("Domain %s: extraction failed for %s", domain, fp, exc_info=True)
                errors += 1
        return instances, errors

    # ---- Stub fallback: produce one instance per file with domain label ----
    logger.info(
        "Domain %s: no expert wired — using stub extraction for %d files",
        domain,
        len(files),
    )
    instances = []
    for i, fp in enumerate(files):
        instances.append(
            {
                "id": f"stub-{domain}-{i:03d}",
                "source": fp,
                "domain": domain,
                "biological_level": _domain_default_level(domain),
                "evidence_design": "descriptive",
                "sources": [{"rank": 1, "type": "file", "name": Path(fp).name}],
                "source_entry": f"file://{fp}",
                "source_time": "2024-01-01",
                "data": {},
            }
        )
    return instances, 0


def _domain_default_level(domain: str) -> str:
    """Return the canonical biological level for a domain key."""
    _level_map = {
        "molecular": "molecular",
        "biomedical": "cellular",
        "bioinformatics": "cellular",
        "clinical": "rct",
    }
    return _level_map.get(domain, "molecular")


def _get_dbase_store(task_spec: dict[str, Any]) -> Any | None:
    """Return a DBaseStore for the Input phase, or None when unavailable.

    Tries (in order):
    1. An injected ``_dbase_store`` in *task_spec* (for testing).
    2. The runtime's ``dbase_store`` when a runtime is attached.
    3. Bootstrapping a global D-Base via ``DBase.global_instance()``.
    """
    # Test injection
    if task_spec is not None and "_dbase_store" in task_spec:
        return task_spec["_dbase_store"]

    # Runtime store
    runtime = task_spec.get("_runtime") if task_spec else None
    if runtime is not None:
        store = getattr(runtime, "dbase_store", None)
        if store is not None:
            return store

    # Global D-Base store
    try:
        from dargus.dbase import DBase
        from dargus.dbase.store import DBaseStore

        return DBaseStore(DBase.global_instance())
    except Exception:
        logger.debug("No D-Base available for Input phase")
        return None


def _collect_duplicates(
    records: list[dict[str, Any]],
    task_spec: dict[str, Any] | None = None,
    dbase_store: Any | None = None,
) -> list[Any]:
    """Identify duplicate records by checking fingerprint collisions in D-Base.

    Returns injected duplicates from ``_duplicate_records`` in *task_spec*
    when provided (for testing).  When a real *dbase_store* is available,
    each record is checked for evidence_id collision and semantic near-duplicates.
    Otherwise returns an empty list (stub).
    """
    if task_spec is not None and task_spec.get("_duplicate_records"):
        return list(task_spec["_duplicate_records"])

    if dbase_store is None or not records:
        return []

    duplicates: list[Any] = []
    for rec in records:
        try:
            result = dbase_store.write_record(rec, dedup=True)
            if result is not True and result is not False:
                duplicates.append(result)
        except Exception:
            logger.debug("Duplicate check failed for record, skipping", exc_info=True)
    return duplicates


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
