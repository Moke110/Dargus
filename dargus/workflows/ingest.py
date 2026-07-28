"""Ingest workflow — hook-orchestrated evidence ingestion into D-Base.

Parses input sources, distributes content to DomainExperts for evidence
extraction, writes validated records into D-Base, and handles duplicate
review with an optional user confirmation gate.
"""

from __future__ import annotations

import json
import logging
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
# Explore-phase constants
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "molecular": [
        "molecule",
        "molecular",
        "compound",
        "ligand",
        "binding",
        "docking",
        "smiles",
        "mol",
        "sdf",
        "pdb",
    ],
    "biomedical": [
        "biomed",
        "biomedical",
        "preclinical",
        "pharmacology",
        "animal",
        "mouse",
        "rat",
        "cell",
        "cancer",
        "tumor",
        "pk",
        "pd",
        "pharmacokinetic",
    ],
    "bioinformatics": [
        "bioinfo",
        "bioinformatics",
        "rna",
        "rna-seq",
        "rnaseq",
        "genomic",
        "proteomic",
        "seq",
        "expression",
        "gene",
        "protein",
        "omics",
        "transcriptom",
    ],
    "clinical": [
        "clinic",
        "clinical",
        "trial",
        "nct",
        "rct",
        "cohort",
        "patient",
        "endpoint",
        "endpoints",
        "phase",
        "dosing",
    ],
}

_VALID_DOMAINS = frozenset({"molecular", "biomedical", "bioinformatics", "clinical"})

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a file classifier for a biomedical evidence ingestion system. "
    "Given a list of filenames, classify each file into exactly one domain: "
    '"molecular", "biomedical", "bioinformatics", "clinical", or "unknown".\n'
    'Return ONLY valid JSON: {"classifications": [{"file": "<name>", "domain": "<domain>"}, ...]}'
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
    2. Explore: scan source_path, classify files by domain via Iris
    3. Distribute content to DomainExperts for evidence extraction
    4. Call dbase_write for each extracted record
    5. Collect DuplicateReviewRequest items
    6. Present duplicates for user confirmation (stub: auto-approve)
    7. Report ingestion summary

    Args:
        task_spec: Dict with keys ``workflow`` (must be ``"ingest"``),
            ``source_path``, optional ``source_type``, ``max_rounds``,
            ``require_confirmation``, ``_reasoning_llm`` (test injection).

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

    # ---- Explore phase: directory scan + domain classification ----------------
    source_path = task_spec.get("source_path", "")
    logger.info("Ingesting from source: %s", source_path)

    reasoning_llm = task_spec.get("_reasoning_llm")
    explore_batches = _explore_sources(source_path, reasoning_llm=reasoning_llm)

    # Preserve domain batches in the session for downstream phases
    if isinstance(ctx.session, dict):
        ctx.session["explore_batches"] = explore_batches

    # Convert explore_batches {domain: [file_path]} into flat records
    # for backward compat with the existing round loop + _partition_by_domain.
    # Each file is a "record stub" until the Convert phase is fully wired.
    parsed_records: list[dict[str, Any]] = []
    for domain, file_paths in explore_batches.items():
        for fp in file_paths:
            parsed_records.append(
                {"domain": domain, "source_path": fp, "source": source_path, "data": {}}
            )

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
# Explore phase
# ---------------------------------------------------------------------------


def _explore_sources(
    source_path: str,
    reasoning_llm: Any | None = None,
) -> dict[str, list[str]]:
    """Scan *source_path* for evidence files and classify by domain.

    Args:
        source_path: Path to a directory of evidence files.
        reasoning_llm: Optional ReasoningLLM used to classify files; when
            absent, classification falls back to a filename heuristic.

    Returns:
        ``{domain: [file_path, ...]}`` where *domain* is one of
        ``"molecular"``, ``"biomedical"``, ``"bioinformatics"``,
        ``"clinical"``.  Files that cannot be classified are logged and
        skipped — not included in any batch.
    """
    directory = Path(source_path)
    if not directory.exists() or not directory.is_dir():
        logger.info(
            "Explore: source directory does not exist or is not a directory: %s",
            source_path,
        )
        return {}

    # ---- Discover files -------------------------------------------------------
    file_names = sorted(
        [p.name for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]
    )

    if not file_names:
        logger.info("Explore: no files found in %s", source_path)
        return {}

    logger.info("Explore: discovered %d files in %s", len(file_names), source_path)

    # ---- Classify each file ---------------------------------------------------
    if reasoning_llm is not None:
        batches = _classify_via_llm(directory, file_names, reasoning_llm)
    else:
        batches = _classify_via_heuristic(directory, file_names)

    # ---- Log summary ----------------------------------------------------------
    total_classified = sum(len(paths) for paths in batches.values())
    n_skipped = len(file_names) - total_classified
    if n_skipped > 0:
        logger.info(
            "Explore: classified %d files, skipped %d unclassifiable files",
            total_classified,
            n_skipped,
        )

    return batches


def _classify_via_llm(
    directory: Path,
    file_names: list[str],
    reasoning_llm: Any,
) -> dict[str, list[str]]:
    """Use the reasoning LLM to classify each file by domain.

    Args:
        directory: The source directory (resolving relative paths).
        file_names: List of filenames to classify.
        reasoning_llm: A ReasoningLLM instance.

    Returns:
        ``{domain: [full_path, ...]}`` for files the LLM classified
        into a known domain. Files returned with ``"unknown"`` or an
        invalid domain are silently skipped.
    """
    from dargus.models.reasoning import Message

    files_json = json.dumps([{"index": i, "filename": n} for i, n in enumerate(file_names)])
    user_prompt = f"Files:\n{files_json}"

    try:
        response = reasoning_llm.chat(
            [
                Message(role="system", content=_CLASSIFY_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ]
        )
        parsed = json.loads(response.content.strip())
    except Exception:
        logger.exception("Explore: LLM classification failed — falling back to heuristic")
        return _classify_via_heuristic(directory, file_names)

    batches: dict[str, list[str]] = {}
    for entry in parsed.get("classifications", []):
        fname = entry.get("file", "")
        domain = (entry.get("domain") or "").strip().lower()
        if domain not in _VALID_DOMAINS:
            logger.info("Explore: skipping '%s' (LLM classified as '%s')", fname, domain)
            continue
        full_path = str(directory / fname)
        batches.setdefault(domain, []).append(full_path)

    return batches


def _classify_via_heuristic(
    directory: Path,
    file_names: list[str],
) -> dict[str, list[str]]:
    """Classify files by scanning filename against domain keyword lists.

    Each filename is lowercased and checked for tokens from
    ``_DOMAIN_KEYWORDS``. The keyword list with the most matches wins
    (ties are broken by list order). Files with zero matches are skipped.
    """
    batches: dict[str, list[str]] = {}

    for fname in file_names:
        fname_lower = fname.lower()
        best_domain: str | None = None
        best_score = 0

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = _keyword_match_score(fname_lower, keywords)
            if score > best_score:
                best_score = score
                best_domain = domain

        if best_domain is None or best_score == 0:
            logger.info("Explore: skipping '%s' — no domain keyword match", fname)
            continue

        full_path = str(directory / fname)
        batches.setdefault(best_domain, []).append(full_path)
        logger.debug("Explore: classified '%s' → %s (score=%d)", fname, best_domain, best_score)

    return batches


def _keyword_match_score(filename_lower: str, keywords: list[str]) -> int:
    """Count how many unique keywords appear as substrings in *filename_lower*.

    Performs raw substring matching so compound-word keywords like
    "rna-seq" match in filenames like ``rna_seq_data.csv``.
    """
    score = 0
    for kw in keywords:
        if kw in filename_lower:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
