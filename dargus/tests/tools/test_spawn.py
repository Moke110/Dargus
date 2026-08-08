"""Tests for the spawn_expert Tool and model-driven predict (T6/T7/T8)."""

from __future__ import annotations

import json

from dargus.experts.reports import expert_report_from_dict
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


def _keyless_runtime():
    """A runtime whose reasoning_llm is wired but unusable (keyless backend).

    Mirrors bootstrap in a keyless-but-configured env: a LiteLLMBackend is
    built even without a key, so ``reasoning_llm is not None`` while
    ``_llm_available()`` is False.
    """
    from dargus.models.reasoning import LiteLLMBackend, ReasoningLLM

    rt = DargusRuntime()
    rt.reasoning_llm = ReasoningLLM(
        backend=LiteLLMBackend(provider="deepseek", model="deepseek-chat", api_key="")
    )
    return rt


def test_keyless_backend_still_produces_des():
    """T8 regression: a keyless-but-configured backend (reasoning_llm wired,
    no usable key) still consults experts and produces a prediction —
    the stub fallback must gate on _llm_available(), not on 'no LLM'."""
    rt = _keyless_runtime()
    iris = rt.agent_factory.iris()
    assert iris._reasoning_llm is not None  # wired...
    assert iris._llm_available() is False  # ...but unusable

    result = iris.predict(
        drug_ids=["chembl:1"],
        disease_id="MONDO:1",
        endpoints=["IC50"],
        max_rounds=1,
    )
    entry = result["chembl:1"]["MONDO:1"]["IC50"]
    # The stub fallback ran (experts were consulted) even though the gate
    # variable is non-None.
    conv = rt.get_conversation("predict:chembl:1:MONDO:1:IC50", "Iris")
    spawn_msgs = [m for m in conv.messages if m.tool_call and m.tool_call.name == "spawn_expert"]
    assert len(spawn_msgs) == 4
    assert entry["confidence_level"] in ("insufficient_data", "low", "medium", "high")


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


def test_second_spawn_keeps_parent_link():
    """T6 regression: consecutive spawns in one run both link to the same
    parent — the Expert's run must restore the parent session_id."""
    rt = _make_runtime()
    rt.agent_factory.iris()  # lazily wires the spawn tool
    tool = rt._spawn_tool

    rt.current_session_id = "predict:chembl:1:MONDO:1:IC50"
    r1 = tool.execute(expert="molecular", drug="chembl:1", disease="MONDO:1", endpoint="IC50")
    r2 = tool.execute(expert="clinical", drug="chembl:1", disease="MONDO:1", endpoint="IC50")

    c1 = rt.get_conversation(r1["session_id"], "MoleculeExpert")
    c2 = rt.get_conversation(r2["session_id"], "ClinicExpert")
    # Both link back to Iris's predict session, not a stale sub-session.
    assert c1.parent_id == "predict:chembl:1:MONDO:1:IC50"
    assert c2.parent_id == "predict:chembl:1:MONDO:1:IC50"


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
    """T6/T7: a spawn shows up as a Tool Message in Iris's Conversation.

    Drives the no-LLM stub path (reasoning_llm=None), which now records each
    spawn as a Tool Message so the stub path is as inspectable as the
    model-driven one (T8 / code-review fix).
    """
    rt = _make_runtime()
    rt.reasoning_llm = None  # no LLM wired -> Iris takes the stub path
    iris = rt.agent_factory.iris()

    result = iris.predict(
        drug_ids=["chembl:1"],
        disease_id="MONDO:1",
        endpoints=["IC50"],
        max_rounds=1,
    )
    assert "chembl:1" in result

    conv = rt.get_conversation("predict:chembl:1:MONDO:1:IC50", "Iris")
    spawn_msgs = [m for m in conv.messages if m.tool_call and m.tool_call.name == "spawn_expert"]
    # Four domain experts were spawned, each recorded as a Tool Message.
    assert len(spawn_msgs) == 4
    assert {m.tool_call.params["expert"] for m in spawn_msgs} == {
        "molecular",
        "biomedical",
        "bioinformatics",
        "clinical",
    }
    for m in spawn_msgs:
        assert m.tool_result is not None and m.tool_result.error is None


def test_expert_self_serves_evidence_in_its_own_conversation():
    """#95 (SPEC-C): a spawned Expert issues ``dbase_query`` itself and its
    ExpertReport is derived from its own Conversation — the tool no longer
    fetches records on the Expert's behalf (no store.read_records)."""
    rt = _make_runtime()
    rt.reasoning_llm = None  # no LLM -> deterministic self-serve path
    rt.agent_factory.iris()  # wires the spawn tool
    tool = rt._spawn_tool

    result = tool.execute(expert="molecular", drug="chembl:1", disease="MONDO:1", endpoint="IC50")
    assert "error" not in result

    sub_session = result["session_id"]
    expert_conv = rt.get_conversation(sub_session, "MoleculeExpert")
    # The Expert's own Conversation carries a dbase_query round.
    query_msgs = [
        m for m in expert_conv.messages if m.tool_call and m.tool_call.name == "dbase_query"
    ]
    assert query_msgs, "Expert must self-serve evidence via dbase_query"
    for m in query_msgs:
        assert m.tool_result is not None and m.tool_result.error is None

    # The report derives from what the Expert gathered — evidence_id-based
    # findings (or a delegation) keyed to the query's scope.
    report = result["report"]
    assert report["expert"] == "MoleculeExpert"


# ------------------------------------------------------------------
# T8 (#91): model-driven predict + TaskDelegation as synthetic message
# ------------------------------------------------------------------


class _SpawnScriptedLLM:
    """Scripted LLM that emits spawn_expert tool calls then converges."""

    def __init__(self, spawns: list[tuple[str, str, str, str]]):
        self.spawns = list(spawns)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.spawns:
            expert, drug, disease, endpoint = self.spawns.pop(0)
            params = {"expert": expert, "drug": drug, "disease": disease, "endpoint": endpoint}
            content = (
                '{"mode": "predict", "action": "tool_call", '
                f'"tool": "spawn_expert", "params": {json.dumps(params)}}}'
            )
        else:
            content = '{"mode": "predict", "action": "text", "text": "synthesized"}'
        return type("R", (), {"content": content})()


def test_model_driven_predict_emits_spawn_expert():
    """Iris's predict run emits spawn_expert Tool calls and the spawns
    appear as Tool Messages in Iris's Conversation."""
    rt = _make_runtime()
    iris = rt.agent_factory.iris()
    iris._reasoning_llm = _SpawnScriptedLLM(
        [
            ("molecular", "chembl:1", "MONDO:1", "IC50"),
            ("clinical", "chembl:1", "MONDO:1", "IC50"),
        ]
    )

    result = iris.predict(
        drug_ids=["chembl:1"],
        disease_id="MONDO:1",
        endpoints=["IC50"],
        max_rounds=3,
    )
    assert "chembl:1" in result
    entry = result["chembl:1"]["MONDO:1"]["IC50"]
    assert "efficacy_score" in entry
    assert entry["reasoning_mode"] == "Iris-model-driven"

    conv = iris._resolve_conversation({"session_id": "predict:chembl:1:MONDO:1:IC50"})
    spawn_msgs = [m for m in conv.messages if m.tool_call and m.tool_call.name == "spawn_expert"]
    assert len(spawn_msgs) >= 1


def test_task_delegation_surfaces_as_synthetic_message():
    """An ExpertReport carrying a TaskDelegation surfaces to Iris as a
    synthetic Message in Iris's Conversation."""
    rt = _make_runtime()
    iris = rt.agent_factory.iris()

    # Build a report that carries a delegation.
    report = expert_report_from_dict(
        {
            "expert": "ClinicExpert",
            "round": 0,
            "findings": [],
            "confidence": {"low": 0.0, "high": 1.0},
            "delegations": [
                {
                    "target_expert": "MoleculeExpert",
                    "record_ids": ["ev_1"],
                    "reason": "molecular record outside clinic scope",
                }
            ],
        }
    )
    conv = rt.get_conversation("predict:test", "Iris")
    iris._surface_delegations(conv, {"ClinicExpert": [report]})

    synthetic = [m for m in conv.messages if m.role == "synthetic"]
    assert len(synthetic) == 1
    assert "MoleculeExpert" in synthetic[0].text
    assert "ev_1" in synthetic[0].text
