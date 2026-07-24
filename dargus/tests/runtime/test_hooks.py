"""Tests for dargus.runtime.hooks — Hook system (Phase B)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from dargus.runtime.context import RuntimeContext
from dargus.runtime.hooks import (
    AcceptanceGateHook,
    HookContext,
    HookPoint,
    HookRegistry,
    ResultReportHook,
    SafetyNetHook,
    SessionInitHook,
    SkeletonContextHook,
    ToolAuditHook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides: Any) -> HookContext:
    """Build a HookContext with a default RuntimeContext and overrides applied."""
    kwargs: dict[str, Any] = {
        "runtime": RuntimeContext(),
        "task_spec": {},
        "session": None,
        "agent": None,
        "round": 0,
        "trace": None,
        "extra": {},
    }
    kwargs.update(overrides)
    return HookContext(**kwargs)


class _CountingHook:
    """A test hook that appends its *tag* to a shared list each time it runs."""

    def __init__(self, tag: str, calls: list[str]):
        self.tag = tag
        self.calls = calls

    def __call__(self, context: HookContext) -> HookContext:
        self.calls.append(self.tag)
        return context


class _MutatingHook:
    """A test hook that sets ``context.extra[key] = value``."""

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def __call__(self, context: HookContext) -> HookContext:
        context.extra[self.key] = self.value
        return context


class _FailingHook:
    """A test hook that always raises the given exception."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def __call__(self, context: HookContext) -> HookContext:
        raise self.exc


# ---------------------------------------------------------------------------
# B1: HookRegistry
# ---------------------------------------------------------------------------


class TestHookRegistry:
    """Tests for HookRegistry — register, run, clear, list_hooks."""

    def test_register_adds_hook(self):
        registry = HookRegistry()
        hook = _CountingHook("a", [])
        registry.register(HookPoint.ROUND_END, hook)
        assert registry.list_hooks(HookPoint.ROUND_END) == [hook]

    def test_register_skips_duplicate(self):
        registry = HookRegistry()
        hook = _CountingHook("a", [])
        registry.register(HookPoint.ROUND_END, hook)
        registry.register(HookPoint.ROUND_END, hook)
        assert len(registry.list_hooks(HookPoint.ROUND_END)) == 1

    def test_register_same_point_different_hooks(self):
        registry = HookRegistry()
        calls: list[str] = []
        h1 = _CountingHook("a", calls)
        h2 = _CountingHook("b", calls)
        registry.register(HookPoint.ROUND_END, h1)
        registry.register(HookPoint.ROUND_END, h2)
        assert len(registry.list_hooks(HookPoint.ROUND_END)) == 2

    def test_run_executes_in_registration_order(self):
        registry = HookRegistry()
        calls: list[str] = []
        registry.register(HookPoint.ROUND_END, _CountingHook("first", calls))
        registry.register(HookPoint.ROUND_END, _CountingHook("second", calls))
        ctx = _make_ctx()
        registry.run(HookPoint.ROUND_END, ctx)
        assert calls == ["first", "second"]

    def test_run_chains_context(self):
        registry = HookRegistry()
        registry.register(HookPoint.ROUND_END, _MutatingHook("a", 1))
        registry.register(HookPoint.ROUND_END, _MutatingHook("b", 2))
        ctx = _make_ctx()
        result = registry.run(HookPoint.ROUND_END, ctx)
        assert result.extra["a"] == 1
        assert result.extra["b"] == 2

    def test_run_no_hooks_returns_context_unchanged(self):
        registry = HookRegistry()
        ctx = _make_ctx(round=5)
        result = registry.run(HookPoint.ROUND_END, ctx)
        assert result is ctx
        assert result.round == 5

    def test_run_wraps_exception_in_runtime_error(self):
        registry = HookRegistry()
        registry.register(HookPoint.ACT_END, _FailingHook(ValueError("boom")))
        ctx = _make_ctx()
        with pytest.raises(RuntimeError, match="failed at point ACT_END"):
            registry.run(HookPoint.ACT_END, ctx)

    def test_run_exception_stops_chain(self):
        """Subsequent hooks must NOT execute after a failure."""
        registry = HookRegistry()
        calls: list[str] = []
        registry.register(HookPoint.ROUND_END, _FailingHook(ValueError("fail")))
        registry.register(HookPoint.ROUND_END, _CountingHook("never", calls))
        ctx = _make_ctx()
        with pytest.raises(RuntimeError):
            registry.run(HookPoint.ROUND_END, ctx)
        assert calls == []

    def test_clear_removes_all_hooks(self):
        registry = HookRegistry()
        registry.register(HookPoint.SESSION_START, _CountingHook("x", []))
        registry.register(HookPoint.SESSION_END, _CountingHook("y", []))
        registry.clear()
        assert registry.list_hooks() == []

    def test_list_hooks_specific_point(self):
        registry = HookRegistry()
        h1 = _CountingHook("a", [])
        h2 = _CountingHook("b", [])
        registry.register(HookPoint.SESSION_START, h1)
        registry.register(HookPoint.SESSION_END, h2)
        assert registry.list_hooks(HookPoint.SESSION_START) == [h1]

    def test_list_hooks_all(self):
        registry = HookRegistry()
        h1 = _CountingHook("a", [])
        h2 = _CountingHook("b", [])
        registry.register(HookPoint.SESSION_START, h1)
        registry.register(HookPoint.SESSION_END, h2)
        all_hooks = registry.list_hooks()
        assert h1 in all_hooks
        assert h2 in all_hooks
        assert len(all_hooks) == 2

    def test_multiple_hooks_same_point_execute_in_registration_order(self):
        registry = HookRegistry()
        calls: list[str] = []
        registry.register(HookPoint.PERCEIVE_START, _CountingHook("1", calls))
        registry.register(HookPoint.PERCEIVE_START, _CountingHook("2", calls))
        registry.register(HookPoint.PERCEIVE_START, _CountingHook("3", calls))
        registry.run(HookPoint.PERCEIVE_START, _make_ctx())
        assert calls == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# B2: SessionInitHook
# ---------------------------------------------------------------------------


class TestSessionInitHook:
    """Tests for SessionInitHook."""

    def test_valid_predict_workflow(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={"workflow": "predict"})
        result = hook(ctx)
        assert result.session is not None
        assert result.session["workflow"] == "predict"
        assert result.session["status"] == "initialized"
        assert "started_at" in result.session

    def test_valid_ingest_workflow(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={"workflow": "ingest"})
        result = hook(ctx)
        assert result.session["workflow"] == "ingest"

    def test_valid_benchmark_workflow(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={"workflow": "benchmark"})
        result = hook(ctx)
        assert result.session["workflow"] == "benchmark"

    def test_session_has_expected_structure(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={"workflow": "predict"})
        result = hook(ctx)
        s = result.session
        assert isinstance(s, dict)
        assert s["workflow"] == "predict"
        assert s["status"] == "initialized"
        assert isinstance(s["started_at"], str)
        assert s["rounds"] == []
        assert s["confirmations"] == []

    def test_missing_workflow_key_raises_valueerror(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={})
        with pytest.raises(ValueError, match="must contain a 'workflow' key"):
            hook(ctx)

    def test_invalid_workflow_raises_valueerror(self):
        hook = SessionInitHook()
        ctx = _make_ctx(task_spec={"workflow": "garbage"})
        with pytest.raises(ValueError, match="Invalid workflow"):
            hook(ctx)

    def test_empty_task_spec_raises_valueerror(self):
        hook = SessionInitHook()
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="must contain a 'workflow' key"):
            hook(ctx)


# ---------------------------------------------------------------------------
# B3: SkeletonContextHook
# ---------------------------------------------------------------------------


class TestSkeletonContextHook:
    """Tests for SkeletonContextHook."""

    def test_injects_fields(self):
        hook = SkeletonContextHook(max_rounds=5)
        ctx = _make_ctx(round=2, extra={"existing": 1})
        result = hook(ctx)
        assert result.extra["round"] == 2
        assert result.extra["max_rounds"] == 5
        assert "elapsed_ms" in result.extra
        assert result.extra["evidence_coverage"] == 0.0
        assert result.extra["pending_delegations"] == 0
        assert result.extra["existing"] == 1  # pre-existing preserved

    def test_uses_constructor_defaults(self):
        hook = SkeletonContextHook()  # default max_rounds=10
        ctx = _make_ctx()
        result = hook(ctx)
        assert result.extra["max_rounds"] == 10

    def test_custom_max_rounds(self):
        hook = SkeletonContextHook(max_rounds=42)
        ctx = _make_ctx()
        result = hook(ctx)
        assert result.extra["max_rounds"] == 42

    def test_elapsed_ms_zero_without_session(self):
        hook = SkeletonContextHook()
        ctx = _make_ctx(session=None)
        result = hook(ctx)
        assert result.extra["elapsed_ms"] == 0.0

    def test_elapsed_ms_computed_from_session(self):
        hook = SkeletonContextHook()
        started = datetime.now(timezone.utc).isoformat()
        ctx = _make_ctx(session={"started_at": started})
        result = hook(ctx)
        assert result.extra["elapsed_ms"] >= 0.0

    def test_preserves_existing_evidence_coverage(self):
        hook = SkeletonContextHook()
        ctx = _make_ctx(extra={"evidence_coverage": 0.75})
        result = hook(ctx)
        assert result.extra["evidence_coverage"] == 0.75

    def test_preserves_existing_pending_delegations(self):
        hook = SkeletonContextHook()
        ctx = _make_ctx(extra={"pending_delegations": 3})
        result = hook(ctx)
        assert result.extra["pending_delegations"] == 3


# ---------------------------------------------------------------------------
# B4: ToolAuditHook
# ---------------------------------------------------------------------------


class _FakeTrace:
    """Minimal fake trace object for testing ToolAuditHook."""

    def __init__(self, tool_name: str | None = None):
        self.tool_name = tool_name


class TestToolAuditHook:
    """Tests for ToolAuditHook."""

    def test_records_tool_call(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log)
        trace = _FakeTrace(tool_name="search_pubmed")
        ctx = _make_ctx(round=3, trace=trace)
        result = hook(ctx)
        assert result is ctx  # returns same context
        assert len(audit_log) == 1
        assert audit_log[0]["round"] == 3
        assert audit_log[0]["tool"] == "search_pubmed"
        assert "timestamp" in audit_log[0]

    def test_noop_when_trace_is_none(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log)
        ctx = _make_ctx(trace=None)
        result = hook(ctx)
        assert result is ctx
        assert audit_log == []

    def test_noop_when_trace_is_none_does_not_crash(self):
        """Sanity check that missing trace attribute on context doesn't crash."""
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log)
        ctx = _make_ctx()  # trace defaults to None
        result = hook(ctx)
        assert result is ctx
        assert audit_log == []

    def test_blocks_disallowed_tool(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log, allowed_tools={"safe_tool"})
        trace = _FakeTrace(tool_name="evil_tool")
        ctx = _make_ctx(trace=trace)
        with pytest.raises(PermissionError, match="not in the allowlist"):
            hook(ctx)
        # blocked entry must still be recorded
        assert len(audit_log) == 1
        assert audit_log[0] == {"status": "blocked", "tool": "evil_tool"}

    def test_allows_tool_in_allowlist(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log, allowed_tools={"safe_tool"})
        trace = _FakeTrace(tool_name="safe_tool")
        ctx = _make_ctx(trace=trace)
        result = hook(ctx)
        assert result is ctx
        assert len(audit_log) == 1
        assert audit_log[0]["tool"] == "safe_tool"
        assert "status" not in audit_log[0]  # not blocked

    def test_no_allowlist_allows_all(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log)  # allowed_tools=None
        trace = _FakeTrace(tool_name="anything")
        ctx = _make_ctx(trace=trace)
        result = hook(ctx)
        assert result is ctx
        assert len(audit_log) == 1
        assert audit_log[0]["tool"] == "anything"

    def test_empty_allowlist_blocks_all(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log, allowed_tools=set())
        trace = _FakeTrace(tool_name="any_tool")
        ctx = _make_ctx(trace=trace)
        with pytest.raises(PermissionError):
            hook(ctx)
        assert audit_log[0] == {"status": "blocked", "tool": "any_tool"}

    def test_trace_without_tool_name_attribute(self):
        """Trace object that lacks a tool_name attribute — should not crash."""
        audit_log: list[dict[str, Any]] = []

        class BareTrace:
            pass

        hook = ToolAuditHook(audit_log=audit_log)
        ctx = _make_ctx(round=1, trace=BareTrace())
        result = hook(ctx)
        assert result is ctx
        # Still records round and timestamp
        assert len(audit_log) == 1
        assert audit_log[0]["round"] == 1

    def test_shared_audit_log_collects_from_multiple_calls(self):
        audit_log: list[dict[str, Any]] = []
        hook = ToolAuditHook(audit_log=audit_log)
        for i in range(3):
            hook(_make_ctx(round=i, trace=_FakeTrace(tool_name=f"tool_{i}")))
        assert len(audit_log) == 3
        assert [e["round"] for e in audit_log] == [0, 1, 2]


# ---------------------------------------------------------------------------
# B5: SafetyNetHook
# ---------------------------------------------------------------------------


class TestSafetyNetHook:
    """Tests for SafetyNetHook."""

    def test_force_converge_after_max_rounds(self):
        hook = SafetyNetHook(max_rounds=3)
        ctx = _make_ctx(round=3)  # round >= max_rounds
        result = hook(ctx)
        assert result.extra["force_converge"] is True

    def test_no_force_converge_before_max_rounds(self):
        hook = SafetyNetHook(max_rounds=5)
        ctx = _make_ctx(round=3)
        result = hook(ctx)
        assert "force_converge" not in result.extra

    def test_timeout_check(self):
        hook = SafetyNetHook(timeout_seconds=0.0, max_rounds=999)
        started = datetime.now(timezone.utc).isoformat()
        ctx = _make_ctx(round=1, session={"started_at": started})
        result = hook(ctx)
        # elapsed > 0 > timeout_seconds(0)
        assert result.extra["force_converge"] is True

    def test_does_not_raise(self):
        """SafetyNetHook must never raise — it always returns the context."""
        hook = SafetyNetHook()
        ctx = _make_ctx(round=999)
        result = hook(ctx)  # should not raise
        assert result is ctx

    def test_insufficient_evidence_after_max_rounds(self):
        hook = SafetyNetHook(max_rounds=3, min_evidence_coverage=0.5)
        ctx = _make_ctx(round=3, extra={"evidence_coverage": 0.1})
        result = hook(ctx)
        assert result.extra.get("insufficient_evidence") is True

    def test_sufficient_evidence_no_flag(self):
        hook = SafetyNetHook(max_rounds=3, min_evidence_coverage=0.5)
        ctx = _make_ctx(round=3, extra={"evidence_coverage": 0.8})
        result = hook(ctx)
        assert "insufficient_evidence" not in result.extra

    def test_evidence_check_only_after_max_rounds(self):
        hook = SafetyNetHook(max_rounds=5, min_evidence_coverage=0.5)
        ctx = _make_ctx(round=2, extra={"evidence_coverage": 0.0})
        result = hook(ctx)
        assert "insufficient_evidence" not in result.extra

    def test_no_session_no_crash(self):
        hook = SafetyNetHook(timeout_seconds=1.0)
        ctx = _make_ctx(round=10, session=None)
        result = hook(ctx)  # should not crash
        assert result.extra["force_converge"] is True

    def test_broken_started_at_does_not_crash(self):
        hook = SafetyNetHook()
        ctx = _make_ctx(round=10, session={"started_at": "not-a-datetime"})
        result = hook(ctx)  # should not crash
        assert result.extra["force_converge"] is True


# ---------------------------------------------------------------------------
# B6: AcceptanceGateHook
# ---------------------------------------------------------------------------


class TestAcceptanceGateHook:
    """Tests for AcceptanceGateHook."""

    def test_valid_report_passes(self):
        hook = AcceptanceGateHook()
        report = {
            "efficacy_low": 0.2,
            "efficacy_up": 0.8,
            "supporting_records": [{"pmid": "123"}],
        }
        ctx = _make_ctx(extra={"FinalReport": report})
        result = hook(ctx)
        assert result is ctx

    def test_noop_when_no_report(self):
        hook = AcceptanceGateHook()
        ctx = _make_ctx()
        result = hook(ctx)
        assert result is ctx

    def test_noop_when_no_report_in_extra_or_session(self):
        hook = AcceptanceGateHook()
        ctx = _make_ctx(extra={"other": 1}, session={"unrelated": 2})
        result = hook(ctx)
        assert result is ctx

    def test_finds_report_in_session(self):
        hook = AcceptanceGateHook()
        report = {"efficacy_low": 0.2, "efficacy_up": 0.8}
        ctx = _make_ctx(session={"FinalReport": report})
        result = hook(ctx)
        assert result is ctx

    def test_report_not_dict_raises(self):
        hook = AcceptanceGateHook()
        ctx = _make_ctx(extra={"FinalReport": "not-a-dict"})
        with pytest.raises(ValueError, match="FinalReport must be a dict"):
            hook(ctx)

    def test_efficacy_low_negative_raises(self):
        hook = AcceptanceGateHook()
        report = {"efficacy_low": -0.1}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="efficacy_low must be in"):
            hook(ctx)

    def test_efficacy_low_above_one_raises(self):
        hook = AcceptanceGateHook()
        report = {"efficacy_low": 1.5}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="efficacy_low must be in"):
            hook(ctx)

    def test_efficacy_up_negative_raises(self):
        hook = AcceptanceGateHook()
        report = {"efficacy_up": -0.5}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="efficacy_up must be in"):
            hook(ctx)

    def test_efficacy_up_above_one_raises(self):
        hook = AcceptanceGateHook()
        report = {"efficacy_up": 2.0}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="efficacy_up must be in"):
            hook(ctx)

    def test_efficacy_low_non_numeric_raises(self):
        hook = AcceptanceGateHook()
        report: dict[str, Any] = {"efficacy_low": "high"}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="efficacy_low must be in"):
            hook(ctx)

    def test_empty_supporting_records_raises(self):
        hook = AcceptanceGateHook()
        report = {"supporting_records": []}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="supporting_records must be a non-empty"):
            hook(ctx)

    def test_supporting_records_wrong_type_raises(self):
        hook = AcceptanceGateHook()
        report: dict[str, Any] = {"supporting_records": "not-a-list"}
        ctx = _make_ctx(extra={"FinalReport": report})
        with pytest.raises(ValueError, match="supporting_records must be a non-empty"):
            hook(ctx)

    def test_missing_both_efficacy_fields_passes(self):
        """Report without efficacy fields is valid (optional fields)."""
        hook = AcceptanceGateHook()
        report: dict[str, Any] = {"supporting_records": [{"a": 1}]}
        ctx = _make_ctx(extra={"FinalReport": report})
        result = hook(ctx)
        assert result is ctx

    def test_boundary_values_pass(self):
        hook = AcceptanceGateHook()
        report = {
            "efficacy_low": 0.0,
            "efficacy_up": 1.0,
            "supporting_records": [{"x": "y"}],
        }
        ctx = _make_ctx(extra={"FinalReport": report})
        result = hook(ctx)
        assert result is ctx


# ---------------------------------------------------------------------------
# B7: ResultReportHook
# ---------------------------------------------------------------------------


class TestResultReportHook:
    """Tests for ResultReportHook."""

    def test_assembles_result_dict(self):
        hook = ResultReportHook()
        ctx = _make_ctx(
            round=5,
            session={"workflow": "predict"},
            extra={"force_converge": True},
        )
        result_ctx = hook(ctx)
        assert "result" in result_ctx.extra
        r = result_ctx.extra["result"]
        assert r["workflow"] == "predict"
        assert r["status"] == "converged"
        assert r["rounds_completed"] == 5

    def test_status_completed_when_no_flags(self):
        hook = ResultReportHook()
        ctx = _make_ctx(round=3, session={"workflow": "ingest"})
        result_ctx = hook(ctx)
        assert result_ctx.extra["result"]["status"] == "completed"

    def test_status_insufficient_evidence(self):
        hook = ResultReportHook()
        ctx = _make_ctx(
            round=10,
            session={"workflow": "predict"},
            extra={"insufficient_evidence": True},
        )
        result_ctx = hook(ctx)
        assert result_ctx.extra["result"]["status"] == "insufficient_evidence"

    def test_workflow_none_when_session_is_none(self):
        hook = ResultReportHook()
        ctx = _make_ctx(round=1, session=None)
        result_ctx = hook(ctx)
        assert result_ctx.extra["result"]["workflow"] is None

    def test_workflow_none_when_session_not_dict(self):
        hook = ResultReportHook()
        ctx = _make_ctx(round=1, session="not-a-dict")
        result_ctx = hook(ctx)
        assert result_ctx.extra["result"]["workflow"] is None


# ---------------------------------------------------------------------------
# B8: Integration / chain tests
# ---------------------------------------------------------------------------


class TestHookChainIntegration:
    """Integration tests: ordering, chain interruption, multiple hooks."""

    def test_registry_runs_multiple_hook_types_in_sequence(self):
        """Full pipeline: SESSION_START -> PERCEIVE_START -> ROUND_END -> SESSION_END."""
        registry = HookRegistry()

        registry.register(HookPoint.SESSION_START, SessionInitHook())
        registry.register(HookPoint.PERCEIVE_START, SkeletonContextHook(max_rounds=5))
        registry.register(HookPoint.ROUND_END, SafetyNetHook(max_rounds=5))
        registry.register(HookPoint.SESSION_END, ResultReportHook())

        ctx = _make_ctx(task_spec={"workflow": "benchmark"})

        # SESSION_START
        ctx = registry.run(HookPoint.SESSION_START, ctx)
        assert ctx.session is not None
        assert ctx.session["workflow"] == "benchmark"

        # PERCEIVE_START
        ctx = registry.run(HookPoint.PERCEIVE_START, ctx)
        assert ctx.extra["max_rounds"] == 5
        assert ctx.extra["evidence_coverage"] == 0.0

        # Several rounds
        for r in range(6):
            ctx.round = r
            ctx = registry.run(HookPoint.ROUND_END, ctx)

        # After max_rounds exceeded, force_converge should be set
        assert ctx.extra.get("force_converge") is True

        # SESSION_END
        ctx = registry.run(HookPoint.SESSION_END, ctx)
        assert "result" in ctx.extra
        assert ctx.extra["result"]["rounds_completed"] == 5

    def test_exception_interrupts_chain_in_registry(self):
        """Hook exception stops subsequent hooks in the same run."""
        registry = HookRegistry()
        calls: list[str] = []
        registry.register(HookPoint.SESSION_START, _CountingHook("first", calls))
        registry.register(HookPoint.SESSION_START, _FailingHook(ValueError("stop")))
        registry.register(HookPoint.SESSION_START, _CountingHook("never", calls))
        ctx = _make_ctx()
        with pytest.raises(RuntimeError):
            registry.run(HookPoint.SESSION_START, ctx)
        assert calls == ["first"]

    def test_session_init_then_acceptance_then_result(self):
        """AcceptanceGateHook (before ResultReportHook) validates, then result is built."""
        registry = HookRegistry()
        registry.register(HookPoint.SESSION_START, SessionInitHook())
        registry.register(HookPoint.SESSION_END, AcceptanceGateHook())
        registry.register(HookPoint.SESSION_END, ResultReportHook())

        ctx = _make_ctx(task_spec={"workflow": "predict"})
        ctx = registry.run(HookPoint.SESSION_START, ctx)

        # Set a valid report
        ctx.extra["FinalReport"] = {
            "efficacy_low": 0.3,
            "efficacy_up": 0.9,
            "supporting_records": [{"id": 1}],
        }
        ctx = registry.run(HookPoint.SESSION_END, ctx)
        assert "result" in ctx.extra
        assert ctx.extra["result"]["status"] == "completed"

    def test_acceptance_gate_blocks_result_when_report_invalid(self):
        """If AcceptanceGateHook raises, ResultReportHook should not run."""
        registry = HookRegistry()
        calls: list[str] = []

        class _TaggedResultHook:
            def __call__(self, context: HookContext) -> HookContext:
                calls.append("result_hook")
                context.extra["result"] = {}
                return context

        registry.register(HookPoint.SESSION_END, AcceptanceGateHook())
        registry.register(HookPoint.SESSION_END, _TaggedResultHook())

        ctx = _make_ctx(extra={"FinalReport": {"efficacy_low": 2.0}})  # invalid
        with pytest.raises(RuntimeError):
            registry.run(HookPoint.SESSION_END, ctx)
        assert "result_hook" not in calls  # never reached

    def test_tool_audit_then_safety_net_chain(self):
        registry = HookRegistry()
        audit_log: list[dict[str, Any]] = []

        registry.register(HookPoint.ACT_END, ToolAuditHook(audit_log=audit_log))
        registry.register(HookPoint.ROUND_END, SafetyNetHook(max_rounds=2))

        trace = _FakeTrace(tool_name="search_pubmed")
        ctx = _make_ctx(round=0, trace=trace)

        # ACT_END
        ctx = registry.run(HookPoint.ACT_END, ctx)
        assert len(audit_log) == 1

        # ROUND_END
        ctx = registry.run(HookPoint.ROUND_END, ctx)
        assert "force_converge" not in ctx.extra  # round 0 < 2

    def test_safety_net_does_not_raise_in_registry(self):
        """Even though SafetyNetHook never raises, the registry must handle it
        correctly when mixed with other hooks."""
        registry = HookRegistry()
        registry.register(HookPoint.ROUND_END, SafetyNetHook(max_rounds=1))
        ctx = _make_ctx(round=5)
        result = registry.run(HookPoint.ROUND_END, ctx)
        assert result.extra["force_converge"] is True


class TestHookProtocolCompliance:
    """Verify that all built-in hooks satisfy the Hook protocol."""

    @pytest.mark.parametrize(
        "hook",
        [
            SessionInitHook(),
            SkeletonContextHook(),
            ToolAuditHook(),
            SafetyNetHook(),
            AcceptanceGateHook(),
            ResultReportHook(),
        ],
    )
    def test_implements_hook_protocol(self, hook: Any) -> None:
        """Each hook must be callable(HookContext) -> HookContext."""
        assert callable(hook)
        ctx = _make_ctx(task_spec={"workflow": "predict"})
        result = hook(ctx)
        assert isinstance(result, HookContext)
