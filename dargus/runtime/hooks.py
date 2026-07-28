"""Hook system — registration, triggering, and execution for Dargus runtime hooks.

Design: ``design/5_hooks.md``. Eleven hook points around the
Perceive → Reason → Act harness and the report flow; six core hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B1: HookPoint, HookContext, Hook Protocol, HookRegistry
# ---------------------------------------------------------------------------


class HookPoint(Enum):
    """Points in the agent loop where hooks can be registered."""

    SESSION_START = auto()
    PERCEIVE_START = auto()
    PERCEIVE_END = auto()
    REASON_START = auto()
    REASON_END = auto()
    ACT_START = auto()
    ACT_END = auto()
    ROUND_END = auto()
    DOMAIN_REPORT_PRODUCED = auto()
    D4_REPORT_PRODUCED = auto()
    SESSION_END = auto()


@dataclass
class HookContext:
    """Context passed to each hook during execution.

    Reserved keys (design/5_hooks.md): ``session``, ``round``, ``agent``,
    ``tools``, ``task_spec``, ``result``, ``error``, ``report_valid``.
    Fields use ``Any`` for forward references to types not yet built
    (WorkflowSession, BaseAgent, CallTrace) so the hook system has no
    dependency on model / agent code.
    """

    runtime: Any  # DargusRuntime (forward ref)
    task_spec: dict[str, Any] = field(default_factory=dict)
    session: Any | None = None  # WorkflowSession (forward ref)
    agent: Any | None = None  # BaseAgent (forward ref)
    tools: dict[str, Any] = field(default_factory=dict)
    round: int = 0
    trace: Any | None = None  # CallTrace (forward ref)
    result: Any | None = None
    error: str | None = None
    report_valid: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class Hook(Protocol):
    """Protocol for hook callables.

    Each hook receives a :class:`HookContext` and returns a (possibly modified)
    :class:`HookContext`.
    """

    def __call__(self, context: HookContext) -> HookContext: ...


class ObserverHook:
    """Wrapper marking a hook observer-only: failures are logged and skipped
    (fail-open) instead of aborting the hook chain."""

    def __init__(self, hook: Hook, name: str | None = None) -> None:
        self.hook = hook
        self.name = name or getattr(hook, "__name__", type(hook).__name__)

    def __call__(self, context: HookContext) -> HookContext:
        return self.hook(context)

    def __repr__(self) -> str:
        return f"ObserverHook({self.name})"


class HookRegistry:
    """Registry that stores and executes hooks keyed by :class:`HookPoint`.

    Enforcement hooks (safety limits and report validation) can never be
    disabled — only advisory core hooks may be named in ``disabled_hooks``.
    """

    #: Hooks whose disablement would bypass a security/enforcement gate.
    ENFORCEMENT_HOOKS = frozenset({"SafetyNetHook", "ReportValidationHook"})

    def __init__(self, disabled_hooks: set[str] | None = None) -> None:
        self._hooks: dict[HookPoint, list[Hook]] = {}
        requested = set(disabled_hooks or set())
        blocked = requested & self.ENFORCEMENT_HOOKS
        if blocked:
            raise ValueError(f"Enforcement hooks cannot be disabled: {sorted(blocked)}")
        self._disabled: set[str] = requested
        # Structured invocation log: hook name, point, timestamp, elapsed, ok
        self.invocation_log: list[dict[str, Any]] = []

    def register(self, point: HookPoint, hook: Hook) -> None:
        """Register *hook* for *point*.  No-op if it is already registered
        at the same point (identity check). Core hooks named in
        ``disabled_hooks`` are skipped.
        """
        name = getattr(hook, "__name__", type(hook).__name__)
        if name in self._disabled:
            logger.info("Hook %s disabled via config — not registered at %s", name, point.name)
            return
        if point not in self._hooks:
            self._hooks[point] = []
        if hook not in self._hooks[point]:
            self._hooks[point].append(hook)

    def run(self, point: HookPoint, context: HookContext) -> HookContext:
        """Run all hooks registered for *point* in registration order.

        Each hook receives the context returned by the previous hook.
        A non-observer hook that raises aborts the chain: the exception is
        wrapped in :class:`RuntimeError` (naming hook and point) and
        re-raised. Observer-only hooks (:class:`ObserverHook`) that raise
        are logged and skipped. Every invocation is recorded in
        :attr:`invocation_log`.
        """
        for hook in self._hooks.get(point, []):
            name = getattr(hook, "__name__", type(hook).__name__)
            observer = isinstance(hook, ObserverHook)
            t0 = time.monotonic()
            ok = True
            error: str | None = None
            try:
                context = hook(context)
            except Exception as exc:
                ok = False
                error = str(exc)
                if observer:
                    logger.warning(
                        "Observer hook %s failed at %s — skipped: %s", name, point.name, exc
                    )
                else:
                    self._log_invocation(name, point, t0, ok, error)
                    raise RuntimeError(
                        f"Hook {hook!r} failed at point {point.name}: {exc}"
                    ) from exc
            self._log_invocation(name, point, t0, ok, error)
        return context

    def _log_invocation(
        self, name: str, point: HookPoint, t0: float, ok: bool, error: str | None
    ) -> None:
        self.invocation_log.append(
            {
                "hook": name,
                "point": point.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.monotonic() - t0) * 1000,
                "ok": ok,
                "error": error,
            }
        )

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._hooks.clear()

    def list_hooks(self, point: HookPoint | None = None) -> list[Hook]:
        """Return hooks registered for *point*, or all hooks if *point* is None."""
        if point is not None:
            return list(self._hooks.get(point, []))
        result: list[Hook] = []
        for hooks in self._hooks.values():
            result.extend(hooks)
        return result


# ---------------------------------------------------------------------------
# B2: SessionInitHook
# ---------------------------------------------------------------------------

_VALID_WORKFLOWS = frozenset({"predict", "ingest"})


class SessionInitHook:
    """Validates ``task_spec`` and initialises a workflow session on
    :attr:`HookPoint.SESSION_START`.
    """

    def __call__(self, context: HookContext) -> HookContext:
        workflow = context.task_spec.get("workflow")
        if workflow is None:
            raise ValueError("task_spec must contain a 'workflow' key")
        if workflow not in _VALID_WORKFLOWS:
            raise ValueError(
                f"Invalid workflow {workflow!r}. " f"Must be one of {sorted(_VALID_WORKFLOWS)}"
            )
        context.session = {
            "workflow": workflow,
            "status": "initialized",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "rounds": [],
            "confirmations": [],
        }
        return context


# ---------------------------------------------------------------------------
# B3: SkeletonContextHook
# ---------------------------------------------------------------------------


class SkeletonContextHook:
    """Injects skeleton context fields into ``context.extra`` on
    :attr:`HookPoint.PERCEIVE_START`.
    """

    def __init__(self, max_rounds: int = 10) -> None:
        self.max_rounds = max_rounds

    def __call__(self, context: HookContext) -> HookContext:
        elapsed_ms = 0.0
        if context.session and "started_at" in context.session:
            try:
                started = datetime.fromisoformat(context.session["started_at"])
                elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
            except (ValueError, TypeError):
                pass

        context.extra["round"] = context.round
        context.extra["max_rounds"] = self.max_rounds
        context.extra["elapsed_ms"] = elapsed_ms
        context.extra.setdefault("evidence_coverage", 0.0)
        context.extra.setdefault("pending_delegations", 0)
        return context


# ---------------------------------------------------------------------------
# B4: ToolAuditHook
# ---------------------------------------------------------------------------


class ToolAuditHook:
    """Audits tool calls on :attr:`HookPoint.ACT_END`.

    Records traces into a shared audit log and optionally blocks tools that
    are not on an explicit allowlist.
    """

    def __init__(
        self,
        audit_log: list[dict[str, Any]] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.audit_log: list[dict[str, Any]] = audit_log if audit_log is not None else []
        self.allowed_tools: set[str] | None = allowed_tools

    def __call__(self, context: HookContext) -> HookContext:
        if context.trace is None:
            return context  # no-op

        entry: dict[str, Any] = {
            "round": context.round,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        tool_name = getattr(context.trace, "tool_name", None)
        if tool_name is not None:
            entry["tool"] = tool_name

        # Permissions check — only enforced when an allowlist is configured.
        if self.allowed_tools is not None and tool_name is not None:
            if tool_name not in self.allowed_tools:
                blocked_entry: dict[str, Any] = {
                    "status": "blocked",
                    "tool": tool_name,
                }
                self.audit_log.append(blocked_entry)
                raise PermissionError(f"Tool {tool_name!r} is not in the allowlist")

        self.audit_log.append(entry)
        return context


# ---------------------------------------------------------------------------
# B5: SafetyNetHook
# ---------------------------------------------------------------------------


class SafetyNetHook:
    """Safety-net guard that runs on :attr:`HookPoint.ROUND_END`.

    Stops the loop (via ``force_converge``) when any limit is reached:
    ``max_rounds`` total rounds, ``round_timeout`` wall-clock per round,
    ``session_timeout`` wall-clock for the whole session. There is no
    minimum-evidence-coverage rule (design/5_hooks.md). Never raises.
    """

    def __init__(
        self,
        max_rounds: int = 10,
        session_timeout: float = 300.0,
        round_timeout: float | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.session_timeout = session_timeout
        self.round_timeout = round_timeout

    def __call__(self, context: HookContext) -> HookContext:
        # Hard round cap
        if context.round >= self.max_rounds:
            context.extra["force_converge"] = True

        # Session timeout (elapsed since session start)
        if context.session and "started_at" in context.session:
            try:
                started = datetime.fromisoformat(context.session["started_at"])
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed > self.session_timeout:
                    context.extra["force_converge"] = True
            except (ValueError, TypeError):
                pass

        # Round timeout (wall-clock of the round that just finished)
        if self.round_timeout is not None:
            round_elapsed_ms = context.extra.get("round_elapsed_ms")
            if round_elapsed_ms is not None and round_elapsed_ms > self.round_timeout * 1000:
                context.extra["force_converge"] = True

        return context


# ---------------------------------------------------------------------------
# B6: ReportValidationHook (+ ReportValidationError)
# ---------------------------------------------------------------------------


class ReportValidationError(ValueError):
    """Raised by :class:`ReportValidationHook` when a report is invalid.

    Carries a structured list of violations (design/5_hooks.md).
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("Report validation failed: " + "; ".join(violations))


class ReportValidationHook:
    """Validates reports on ``DOMAIN_REPORT_PRODUCED``, ``D4_REPORT_PRODUCED``,
    and :attr:`HookPoint.SESSION_END`.

    Checks (design/5_hooks.md §ReportValidationHook):
    1. report format (must be a dict),
    2. presence and valid range of ``efficacy_score`` (DES) and
       ``confidence_score`` (DCS) — waived when ``confidence_level`` is
       ``insufficient_data``, in which case both scores must be unset,
    3. existence in D-Base of every ``evidence_id`` cited in
       ``supporting_records`` (only when a D-Base is wired via ``dbase``).

    On failure: sets ``context.report_valid = False`` and raises
    :class:`ReportValidationError` with the structured violations.
    No-op when no report is present (some sessions do not produce reports).
    """

    def __init__(self, dbase: Any | None = None) -> None:
        self.dbase = dbase

    def __call__(self, context: HookContext) -> HookContext:
        report = context.extra.get("FinalReport")
        if report is None and isinstance(context.session, dict):
            report = context.session.get("FinalReport")

        if report is None:
            return context  # no-op

        violations: list[str] = []

        if not isinstance(report, dict):
            raise ReportValidationError(
                [f"FinalReport must be a dict, got {type(report).__name__}"]
            )

        # Detect shape: nested prediction contract vs flat dict.
        # Nested:  {drug_id: {disease_id: {endpoint: {DES, DCS, ...}}}}
        # Flat:    {"efficacy_score": 0.5, ...} or {"n_records": N, ...} (ingest)
        # Heuristic: if NO top-level key is a known inner key AND at least
        # one top-level value is a dict, treat it as nested.
        # Guard: ingest-style reports (n_records, source_path, etc.) are never nested.
        _INNER_KEYS = frozenset(
            {
                "efficacy_score",
                "confidence_score",
                "supporting_records",
                "confidence_level",
                "reasoning_mode",
            }
        )
        _INGEST_KEYS = frozenset({"n_records", "source_path"})
        has_inner = any(k in _INNER_KEYS for k in report)
        has_dict_val = any(isinstance(v, dict) for v in report.values())
        is_ingest = any(k in _INGEST_KEYS for k in report)

        if not has_inner and has_dict_val and not is_ingest:
            self._validate_nested(report, violations)
        elif has_inner:
            # Flat prediction dict
            self._validate_flat(report, violations)
        # else: flat non-prediction dict (e.g. ingest summary) — skip validation

        if violations:
            context.report_valid = False
            raise ReportValidationError(violations)

        context.report_valid = True
        return context

    # ------------------------------------------------------------------
    # Flat dict validation (backward-compatible)
    # ------------------------------------------------------------------

    def _validate_flat(self, report: dict, violations: list[str]) -> None:
        """Validate a flat FinalReport dict."""
        if report.get("confidence_level") == "insufficient_data":
            for key in ("efficacy_score", "confidence_score"):
                if key in report and report[key] is not None:
                    violations.append(
                        f"{key} must be unset when confidence_level is "
                        f"insufficient_data, got {report[key]!r}"
                    )
        else:
            for key in ("efficacy_score", "confidence_score"):
                if key in report:
                    val = report[key]
                    if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                        violations.append(f"{key} must be in [0, 1], got {val!r}")

        records = report.get("supporting_records")
        if records is not None and report.get("confidence_level") != "insufficient_data":
            if not isinstance(records, list) or len(records) == 0:
                violations.append(f"supporting_records must be a non-empty list, got {records!r}")

        if self.dbase is not None and isinstance(records, list):
            for rid in records:
                if not isinstance(rid, str) or not rid.startswith("ev_"):
                    continue
                if not self.dbase.evidence_id_exists(rid):
                    violations.append(f"supporting record {rid!r} not found in D-Base")

    # ------------------------------------------------------------------
    # Nested contract validation
    # ------------------------------------------------------------------

    def _validate_nested(self, report: dict, violations: list[str]) -> None:
        """Validate the universal nested contract:
        ``{drug_id: {disease_id: {endpoint: {efficacy_score, confidence_score,
        supporting_records, reasoning_mode, confidence_level}}}}``
        """
        if not report:
            violations.append("nested FinalReport must be non-empty")
            return

        for drug_id, diseases in report.items():
            if not isinstance(diseases, dict):
                violations.append(
                    f"expected disease dict under drug {drug_id!r}, "
                    f"got {type(diseases).__name__}"
                )
                continue
            if not diseases:
                violations.append(f"missing disease_id under drug {drug_id!r}")
                continue
            for disease_id, endpoints in diseases.items():
                if not isinstance(endpoints, dict):
                    violations.append(
                        f"expected endpoint dict under {drug_id}/{disease_id}, "
                        f"got {type(endpoints).__name__}"
                    )
                    continue
                if not endpoints:
                    violations.append(f"missing endpoint under {drug_id}/{disease_id}")
                    continue
                for endpoint, entry in endpoints.items():
                    if not isinstance(entry, dict):
                        violations.append(
                            f"expected prediction dict under "
                            f"{drug_id}/{disease_id}/{endpoint}, "
                            f"got {type(entry).__name__}"
                        )
                        continue
                    self._validate_endpoint_entry(
                        entry,
                        f"{drug_id}/{disease_id}/{endpoint}",
                        violations,
                    )

    def _validate_endpoint_entry(
        self,
        entry: dict,
        path: str,
        violations: list[str],
    ) -> None:
        """Validate a single endpoint prediction entry."""
        _REQUIRED_KEYS = {
            "efficacy_score",
            "confidence_score",
            "supporting_records",
            "reasoning_mode",
            "confidence_level",
        }
        missing = _REQUIRED_KEYS - set(entry.keys())
        for mk in sorted(missing):
            violations.append(f"missing {mk} in {path}")

        confidence_level = entry.get("confidence_level")

        # DES ± DCS range (waived for insufficient_data)
        if confidence_level == "insufficient_data":
            for key in ("efficacy_score", "confidence_score"):
                if entry.get(key) is not None:
                    violations.append(
                        f"{path}: {key} must be unset when confidence_level is "
                        f"insufficient_data, got {entry[key]!r}"
                    )
        else:
            for key in ("efficacy_score", "confidence_score"):
                if key in entry:
                    val = entry[key]
                    if not isinstance(val, (int, float, type(None))):
                        violations.append(f"{path}: {key} must be numeric or None, got {val!r}")
                    elif isinstance(val, (int, float)) and not (0 <= val <= 1):
                        violations.append(f"{path}: {key} must be in [0, 1], got {val!r}")
                    elif val is None and confidence_level != "insufficient_data":
                        violations.append(
                            f"{path}: {key} must not be None when "
                            f"confidence_level is {confidence_level!r}"
                        )

        # supporting_records (insufficient_data reports may cite zero records)
        records = entry.get("supporting_records")
        if records is not None and confidence_level != "insufficient_data":
            if not isinstance(records, list) or len(records) == 0:
                violations.append(
                    f"{path}: supporting_records must be a non-empty list, " f"got {records!r}"
                )

        # evidence_id existence in D-Base
        if self.dbase is not None and isinstance(records, list):
            for rid in records:
                if not isinstance(rid, str) or not rid.startswith("ev_"):
                    continue  # non-evidence citation forms are out of scope
                if not self.dbase.evidence_id_exists(rid):
                    violations.append(f"{path}: supporting record {rid!r} not found in D-Base")


# ---------------------------------------------------------------------------
# B7: ResultReportHook
# ---------------------------------------------------------------------------


class ResultReportHook:
    """Assembles a result dict into ``context.extra["result"]`` on
    :attr:`HookPoint.SESSION_END`.

    Expected to run **after** :class:`ReportValidationHook` in the registration
    order so the report has already been validated.
    """

    def __call__(self, context: HookContext) -> HookContext:
        workflow = None
        if isinstance(context.session, dict):
            workflow = context.session.get("workflow")

        status = "completed"
        if context.extra.get("force_converge"):
            status = "converged"
        if context.extra.get("insufficient_evidence"):
            status = "insufficient_evidence"

        result: dict[str, Any] = {
            "workflow": workflow,
            "status": status,
            "rounds_completed": context.round,
        }
        context.extra["result"] = result
        context.result = result
        return context
