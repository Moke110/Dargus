"""Tests for AgentReport and CallTrace dataclasses."""

from dargus.agents.report import AgentReport, CallTrace


def test_call_trace_defaults():
    trace = CallTrace(round=1, phase="planner")
    assert trace.round == 1
    assert trace.phase == "planner"
    assert trace.skill_used is None
    assert trace.tool_called is None
    assert trace.knowledge_retrieved == []
    assert trace.error is None


def test_agent_report_construction():
    report = AgentReport(
        agent_name="TestAgent",
        task_spec={"drug_ids": ["aspirin"]},
        rounds=3,
        converged=True,
        confidence=0.85,
    )
    assert report.agent_name == "TestAgent"
    assert report.findings == []
    assert report.call_trace == []
    assert report.data_gaps == []


def test_agent_report_with_traces():
    traces = [
        CallTrace(round=0, phase="planner", elapsed_ms=150),
        CallTrace(round=0, phase="executor", tool_called="dbase_query", elapsed_ms=200),
        CallTrace(round=0, phase="critic", elapsed_ms=180),
    ]
    report = AgentReport(
        agent_name="Test",
        task_spec={},
        rounds=1,
        converged=False,
        confidence=0.5,
        call_trace=traces,
        data_gaps=["missing_rct"],
    )
    assert len(report.call_trace) == 3
    assert report.data_gaps == ["missing_rct"]
