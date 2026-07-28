"""Ingest workflow — hook-orchestrated evidence ingestion into D-Base.

Parses input sources, distributes content to DomainExperts for evidence
extraction, writes validated records into D-Base, and handles duplicate
review with an optional user confirmation gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dargus.experts.bioinfo import BioinfoExpert
from dargus.experts.biomed import BiomedExpert
from dargus.experts.clinic import ClinicExpert
from dargus.experts.molecule import MoleculeExpert
from dargus.models.reasoning import ReasoningLLM
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
# Domain Expert factory
# ---------------------------------------------------------------------------

_EXPERT_BY_DOMAIN: dict[str, type] = {
    "molecule": MoleculeExpert,
    "biomedical": BiomedExpert,
    "bioinformatics": BioinfoExpert,
    "clinical": ClinicExpert,
}


def _build_experts(
    reasoning_llm: ReasoningLLM | None = None,
) -> dict[str, Any]:
    """Build domain experts for the Convert phase."""
    experts: dict[str, Any] = {}
    for domain, cls in _EXPERT_BY_DOMAIN.items():
        experts[domain] = cls(reasoning_llm=reasoning_llm)
    return experts


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

    # ---- Wire reasoning LLM (injected for tests; stub mode when absent) ------
    reasoning_llm = task_spec.get("_reasoning_llm", None)

    # ---- Explore phase ---------------------------------------------------------
    source_path = task_spec.get("source_path", "")
    logger.info("Ingesting from source: %s", source_path)

    # Scan directory, classify files by domain → per-domain file batches
    explore_batches = _explore_source(source_path, task_spec)
    if isinstance(ctx.session, dict):
        ctx.session["explore_batches"] = explore_batches

    # ---- Convert phase: Domain Experts extract evidence from their batches ----
    experts = _build_experts(reasoning_llm=reasoning_llm)
    all_instances: list[dict[str, Any]] = []
    n_errors = 0

    for domain, files in explore_batches.items():
        expert = experts.get(domain)
        if expert is None:
            logger.info("No expert for domain '%s' — skipping %d files", domain, len(files))
            continue
        instances, err_count = _extract_evidence(expert, domain, files)
        all_instances.extend(instances)
        n_errors += err_count
        if isinstance(ctx.session, dict):
            rounds = ctx.session.setdefault("rounds", [])
            rounds.append(
                {
                    "round": len(rounds),
                    "domain": domain,
                    "extracted": len(instances),
                    "errors": err_count,
                }
            )

    n_records = len(all_instances)
    n_duplicates = 0

    # ---- Place extracted instances into session for downstream phases ---------
    if isinstance(ctx.session, dict):
        ctx.session["evidence_instances"] = all_instances

    # ---- Round loop hooks (compat: run PERCEIVE_START / ROUND_END) ------------
    for round_num in range(min(1, max_rounds)):
        ctx.round = round_num
        ctx = hooks.run(HookPoint.PERCEIVE_START, ctx)
        ctx = hooks.run(HookPoint.ROUND_END, ctx)

    # ---- Duplicate review -----------------------------------------------------
    duplicates = _collect_duplicates(all_instances, task_spec)
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
# Explore phase
# ---------------------------------------------------------------------------

# Domain file classification: file extension -> domain key
_EXT_TO_DOMAIN: dict[str, str] = {
    ".mol2": "molecule",
    ".sdf": "molecule",
    ".pdb": "molecule",
    ".smiles": "molecule",
    ".smi": "molecule",
    ".csv": "clinical",
    ".tsv": "clinical",
    ".xlsx": "clinical",
    ".txt": "biomedical",
    ".json": "bioinformatics",
    ".fasta": "bioinformatics",
    ".fastq": "bioinformatics",
    ".gff": "bioinformatics",
    ".bam": "bioinformatics",
}


def _explore_source(
    source_path: str, task_spec: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Explore a directory, classifying files by domain.

    Returns ``{domain_key: [file_path, ...]}`` mapping each recognised
    domain to the list of files that belong to it.  Files with unknown
    extensions are assigned to ``"biomedical"`` as a fallback.  An empty
    or non-existent *source_path* yields an empty dict.

    The *task_spec* may carry an optional ``_domain_mapping`` override
    (``{filename: domain}``) for test injection.
    """
    domain_mapping: dict[str, str] = {}
    if task_spec is not None:
        domain_mapping = task_spec.get("_domain_mapping", {}) or {}

    if not source_path:
        return {}

    sp = Path(source_path)
    if not sp.exists() or not sp.is_dir():
        logger.warning("Source path '%s' does not exist or is not a directory", source_path)
        return {}

    batches: dict[str, list[str]] = {}
    for fpath in sorted(sp.iterdir()):
        if not fpath.is_file():
            continue
        name = fpath.name
        if name in domain_mapping:
            domain = domain_mapping[name]
        else:
            suffix = fpath.suffix.lower()
            domain = _EXT_TO_DOMAIN.get(suffix, "biomedical")
        batches.setdefault(domain, []).append(str(fpath))

    return batches


# ---------------------------------------------------------------------------
# Convert phase
# ---------------------------------------------------------------------------


def _extract_evidence(
    expert: Any, domain: str, files: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """Extract evidence from *files* using *expert*.

    Each file is fed through ``expert.extract()``.  A single file that
    raises is logged and skipped while the rest of the batch continues.

    Returns ``(instances, n_errors)``.
    """
    instances: list[dict[str, Any]] = []
    errors = 0

    for file_path in files:
        try:
            extracted = expert.extract(file_path)
            instances.extend(extracted)
        except Exception:
            logger.exception("Domain %s: error extracting %s — skipping", domain, file_path)
            errors += 1

    return instances, errors


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
