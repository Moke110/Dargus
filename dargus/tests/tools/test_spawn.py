"""Tests for the spawn_expert Tool (T6 / #89, SPEC-C)."""

from __future__ import annotations

from dargus.runtime.context import DargusRuntime


class _FakeLLM:
    """Stub reasoning LLM — returns a terminal text response."""

    def chat(self, messages):
        content = '{"mode": "predict", "action": "text", "text": "assessed"}'
        return type("R", (), {"content": content})()


def _make_runtime():
    rt = DargusRuntime()
    rt.reasoning_llm = _FakeLLM()
    return rt


def test_spawn_expert_registered_in_predict_mode():
    """spawn_expert is registered as a Tool and available in predict mode."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    assert rt._spawn_tool is not None
    assert rt._spawn_tool.name == "spawn_expert"
    assert "spawn_expert" in rt.mode_config["predict"].tools
    # Not available in auto/ingest modes.
    assert "spawn_expert" not in rt.mode_config["auto"].tools
    assert "spawn_expert" not in rt.mode_config["ingest"].tools


def test_spawn_expert_returns_expert_report_as_tool_result():
    """Invoking spawn_expert runs the Expert and returns an ExpertReport."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    tool = rt._spawn_tool

    result = tool.execute(
        expert="molecular",
        drug="chembl:1",
        disease="MONDO:1",
        endpoint="IC50",
    )
    assert "expert" in result
    assert result["expert"] == "MoleculeExpert"
    assert "report" in result
    report = result["report"]
    assert report["expert"] == "MoleculeExpert"
    # No D-Base records -> insufficient evidence, but the report is well-formed.
    assert "findings" in report
    assert "confidence" in report
    assert "data_gaps" in report


def test_spawn_expert_links_parent_conversation():
    """The Expert's Conversation is parent_id-linked to Iris's."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    # Simulate Iris running in a session.
    rt.current_session_id = "dialogue"
    tool = rt._spawn_tool

    result = tool.execute(
        expert="clinical",
        drug="chembl:2",
        disease="MONDO:2",
        endpoint="efficacy",
    )
    sub_session = result["session_id"]
    child = rt.get_conversation(sub_session, "ClinicExpert")
    assert child.parent_id == "dialogue"


def test_spawn_expert_depth_guard_denies_subagent():
    """A Subagent (spawn_stack non-empty) is denied by the depth guard."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    tool = rt._spawn_tool

    rt.spawn_stack.append("dialogue:molecular:chembl:1:MONDO:1:IC50")
    try:
        result = tool.execute(
            expert="biomedical",
            drug="chembl:1",
            disease="MONDO:1",
            endpoint="IC50",
        )
    finally:
        rt.spawn_stack.pop()

    assert "error" in result
    assert "depth" in result["error"].lower() or "denied" in result["error"].lower()


def test_spawn_expert_unknown_domain_errors():
    """An unknown expert domain returns an error dict, not a raise."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    tool = rt._spawn_tool
    result = tool.execute(
        expert="cardiology",
        drug="chembl:1",
        disease="MONDO:1",
        endpoint="IC50",
    )
    assert "error" in result


# ------------------------------------------------------------------
# T7 (#90): Expert self-serve + least-privilege
# ------------------------------------------------------------------


def test_expert_mode_is_least_privilege():
    """Experts run in 'expert' mode: dbase_query + read only; NOT
    switch_mode / write_file / spawn_expert."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    expert = rt.agent_factory.expert("molecular")
    assert expert._mode == "expert"

    mode_tools = set(rt.mode_config["expert"].tools)
    assert {"dbase_query", "read_file"} <= mode_tools
    assert "switch_mode" not in mode_tools
    assert "write_file" not in mode_tools
    assert "spawn_expert" not in mode_tools


def test_expert_cannot_invoke_forbidden_tools():
    """An Expert's ACT is denied for switch_mode / write_file / spawn_expert."""
    rt = _make_runtime()
    rt.agent_factory.iris()
    expert = rt.agent_factory.expert("clinical")
    expert._runtime = rt

    # Expert's perceive cache carries the expert-mode tool allowlist.
    perceive = expert._perceive({"workflow": "predict"}, expert._resolve_conversation({}), 0)
    allowed = set(perceive["mode_tool_names"])
    for forbidden in ("switch_mode", "write_file", "spawn_expert"):
        assert forbidden not in allowed

    # ACT rejects a switch_mode call in expert mode.
    out, _traces = expert._act(
        {"action": "tool_call", "tool": "switch_mode", "params": {"target": "auto"}},
        0,
        perceive,
    )
    assert "error" in out["output"]
    assert "not permitted" in out["output"]["error"]


def test_spawn_expert_records_tool_message_in_iris_conversation():
    """T6/T7: a spawn shows up as a Tool Message in Iris's Conversation."""
    rt = _make_runtime()
    iris = rt.agent_factory.iris()
    tool = rt._spawn_tool

    iris.run({"query": "predict aspirin", "session_id": "dialogue"})
    # Manually invoke the spawn tool within Iris's session.
    result = tool.execute(
        expert="molecular",
        drug="chembl:1",
        disease="MONDO:1",
        endpoint="IC50",
    )
    assert "report" in result
