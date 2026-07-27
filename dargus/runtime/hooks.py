"""Hook system — registration, triggering, and execution for Dargus runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# B1: HookPoint, HookContext, Hook Protocol, HookRegistry
# ---------------------------------------------------------------------------


class HookPoint(Enum):
    """Points in the agent loop where hooks can be registered."""

    SESSION_START = auto()
    PERCEIVE_START = auto()
    PLAN_END = auto()
    ACT_END = auto()
    CRITIC_END = auto()
    ROUND_END = auto()
    SESSION_END = auto()


@dataclass
class HookContext:
    """Context passed to each hook during execution.

    Fields use ``Any`` for forward references to types not yet built
    (WorkflowSession, BaseAgent, CallTrace) so the hook system has no
    dependency on model / agent code.
    """

    runtime: Any  # RuntimeContext (forward ref)
    task_spec: dict[str, Any] = field(default_factory=dict)
    session: Any | None = None  # WorkflowSession (forward ref)
    agent: Any | None = None  # BaseAgent (forward ref)
    round: int = 0
    trace: Any | None = None  # CallTrace (forward ref)
    extra: dict[str, Any] = field(default_factory=dict)


class Hook(Protocol):
    """Protocol for hook callables.

    Each hook receives a :class:`HookContext` and returns a (possibly modified)
    :class:`HookContext`.
    """

    def __call__(self, context: HookContext) -> HookContext: ...


class HookRegistry:
    """Registry that stores and executes hooks keyed by :class:`HookPoint`."""

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[Hook]] = {}

    def register(self, point: HookPoint, hook: Hook) -> None:
        """Register *hook* for *point*.  No-op if it is already registered
        at the same point (identity check).
        """
        if point not in self._hooks:
            self._hooks[point] = []
        if hook not in self._hooks[point]:
            self._hooks[point].append(hook)

    def run(self, point: HookPoint, context: HookContext) -> HookContext:
        """Run all hooks registered for *point* in registration order.

        Each hook receives the context returned by the previous hook.
        If any hook raises an exception it is wrapped in :class:`RuntimeError`
        (with a message that names the failing hook and point) and re-raised.
        """
        for hook in self._hooks.get(point, []):
            try:
                context = hook(context)
            except Exception as exc:
                raise RuntimeError(f"Hook {hook!r} failed at point {point.name}: {exc}") from exc
        return context

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

_VALID_WORKFLOWS = frozenset({"predict", "ingest", "benchmark"})


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

    Sets ``force_converge`` and ``insufficient_evidence`` flags in
    ``context.extra`` when safety limits are exceeded.  Never raises.
    """

    def __init__(
        self,
        max_rounds: int = 10,
        timeout_seconds: float = 300.0,
        min_evidence_coverage: float = 0.0,
    ) -> None:
        self.max_rounds = max_rounds
        self.timeout_seconds = timeout_seconds
        self.min_evidence_coverage = min_evidence_coverage

    def __call__(self, context: HookContext) -> HookContext:
        elapsed: float | None = None
        if context.session and "started_at" in context.session:
            try:
                started = datetime.fromisoformat(context.session["started_at"])
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            except (ValueError, TypeError):
                pass

        # Hard round cap
        if context.round >= self.max_rounds:
            context.extra["force_converge"] = True

        # Timeout
        if elapsed is not None and elapsed > self.timeout_seconds:
            context.extra["force_converge"] = True

        # Evidence shortfall after exhausting rounds
        if context.round >= self.max_rounds:
            coverage = context.extra.get("evidence_coverage", 0)
            if coverage < self.min_evidence_coverage:
                context.extra["insufficient_evidence"] = True

        return context


# ---------------------------------------------------------------------------
# B6: AcceptanceGateHook
# ---------------------------------------------------------------------------


class AcceptanceGateHook:
    """Validates the final report on :attr:`HookPoint.SESSION_END`.

    No-op when no report is present (some sessions do not produce reports).
    """

    def __call__(self, context: HookContext) -> HookContext:
        report = context.extra.get("FinalReport")
        if report is None and isinstance(context.session, dict):
            report = context.session.get("FinalReport")

        if report is None:
            return context  # no-op

        if not isinstance(report, dict):
            raise ValueError("FinalReport must be a dict")

        # DES ± DCS: both scores must be present and in [0, 1], except when
        # confidence_level is "insufficient_data" — then both must be unset.
        if report.get("confidence_level") == "insufficient_data":
            for key in ("efficacy_score", "confidence_score"):
                if report.get(key) is not None:
                    raise ValueError(
                        f"{key} must be unset when confidence_level is "
                        f"insufficient_data, got {report[key]!r}"
                    )
        else:
            for key in ("efficacy_score", "confidence_score"):
                if key in report:
                    val = report[key]
                    if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                        raise ValueError(f"{key} must be in [0, 1], got {val!r}")

        # supporting_records (insufficient_data reports may cite zero records)
        if "supporting_records" in report and report.get("confidence_level") != "insufficient_data":
            records = report["supporting_records"]
            if not isinstance(records, list) or len(records) == 0:
                raise ValueError(
                    f"supporting_records must be a non-empty list, " f"got {records!r}"
                )

        return context


# ---------------------------------------------------------------------------
# B7: ResultReportHook
# ---------------------------------------------------------------------------


class ResultReportHook:
    """Assembles a result dict into ``context.extra["result"]`` on
    :attr:`HookPoint.SESSION_END`.

    Expected to run **after** :class:`AcceptanceGateHook` in the registration
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
        return context
