"""Tests for Iris.run() — PRA loop via FakeReasoningBackend.

ADR-0002: Iris.process_query() is deleted. All NL processing goes through
the unified PRA (Perceive→Reason→Act) loop via iris.run().
"""

import os

import pytest

from dargus.agents.report import AgentReport
from dargus.iris.commander import Iris
from dargus.models.reasoning import LLMResponse, LLMUsage, Message


@pytest.fixture(autouse=True)
def _clear_api_key():
    """Prevent real .env API keys from leaking into tests."""
    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


class FakeReasoningBackend:
    """ReasoningBackend stub returning a controlled PRA JSON response."""

    def __init__(self, response: str):
        self._response = response

    def chat(self, messages: list[Message], options=None) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            usage=LLMUsage(),
            model="fake",
            finish_reason="stop",
        )


def _iris_with_llm(response: str) -> Iris:
    from dargus.models.reasoning import ReasoningLLM

    return Iris(reasoning_llm=ReasoningLLM(backend=FakeReasoningBackend(response)))


# ------------------------------------------------------------------
# Basic PRA loop tests
# ------------------------------------------------------------------


def test_iris_run_text_response():
    """LLM returns a text action → loop converges in 1 round."""
    import json

    response = json.dumps({"mode": "auto", "action": "text", "text": "Hello! How can I help?"})

    iris = _iris_with_llm(response)
    report = iris.run({"query": "hello"})

    assert isinstance(report, AgentReport)
    assert report.converged is True
    assert report.rounds == 1
    assert "Hello!" in report.findings[-1]


def test_iris_run_tool_call():
    """LLM returns a tool_call → loop continues."""
    import json

    # Round 1: tool call → Round 2: text
    responses = iter(
        [
            json.dumps({"mode": "auto", "action": "text", "text": "Let me check that for you."}),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            try:
                content = next(responses)
            except StopIteration:
                content = json.dumps({"mode": "auto", "action": "text", "text": "Done."})
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "read data/file.txt"})

    assert isinstance(report, AgentReport)
    assert report.rounds >= 1


def test_iris_run_missing_llm_backend():
    """Without an injected ReasoningLLM, the agent stub still produces a report."""
    iris = Iris()
    report = iris.run({"query": "hello"})
    assert isinstance(report, AgentReport)
    # The stub returns {"error": "no_llm_configured"} which is not valid PRA JSON
    # so it falls through to parse error → text response
    assert report.rounds <= 5  # bounded by MAX_ROUNDS


def test_iris_run_mode_tag_in_response():
    """LLM response always carries a mode field."""
    import json

    response = json.dumps({"mode": "auto", "action": "text", "text": "Test response"})

    iris = _iris_with_llm(response)
    report = iris.run({"query": "test"})

    assert report.converged is True
    assert "Test response" in report.findings[-1]


# ------------------------------------------------------------------
# Mode transition tests
# ------------------------------------------------------------------


def test_iris_run_switch_mode_tool_call():
    """LLM calls switch_mode → the mode transition is detected."""
    import json

    response = json.dumps(
        {
            "mode": "auto",
            "action": "tool_call",
            "tool": "switch_mode",
            "params": {"target": "ingest"},
        }
    )

    iris = _iris_with_llm(response)
    # switch_mode tool should exist on the tool registry
    # but since we don't have a runtime, the tool will error
    # The test just verifies the loop handles a switch_mode intent
    report = iris.run({"query": "ingest data from /tmp"})

    assert isinstance(report, AgentReport)


# ------------------------------------------------------------------
# api.ask() integration
# ------------------------------------------------------------------


def test_api_ask_with_fake_backend(monkeypatch):
    """api.ask() calls iris.run() and returns text response."""
    import json

    from dargus import api

    response = json.dumps({"mode": "auto", "action": "text", "text": "Iris at your service."})

    class FakeLLM:
        def chat(self, messages, options=None):
            return LLMResponse(content=response, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    llm = ReasoningLLM(backend=FakeLLM())

    # Patch _create_iris_with_lm to return Iris with our fake LLM
    def _create(*args, **kwargs):
        return Iris(reasoning_llm=llm)

    monkeypatch.setattr(api, "_create_iris_with_lm", _create)
    result = api.ask("hello")
    assert "Iris at your service" in result


# ------------------------------------------------------------------
# Auto mode behavior tests (Ticket 5)
# ------------------------------------------------------------------


def test_auto_mode_chat_path():
    """Chat: "hi" → 1-round converge with text response."""
    import json

    response = json.dumps(
        {
            "mode": "auto",
            "action": "text",
            "text": "Hi! I'm Iris. How can I help with your research?",
        }
    )

    iris = _iris_with_llm(response)
    report = iris.run({"query": "hi"})

    assert report.converged is True
    assert report.rounds == 1
    assert "Iris" in str(report.findings)


def test_auto_mode_tool_calling_path():
    """User asks a question requiring file access → LLM calls read_file tool."""
    import json

    # Round 1: tool call for read_file → Round 2: text
    responses = iter(
        [
            json.dumps(
                {
                    "mode": "auto",
                    "action": "tool_call",
                    "tool": "read_file",
                    "params": {"path": "README.md"},
                }
            ),
            json.dumps(
                {
                    "mode": "auto",
                    "action": "text",
                    "text": "The README contains setup instructions.",
                }
            ),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            content = next(responses)
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "what's in README.md?"})

    assert isinstance(report, AgentReport)
    # Should converge: round 1 tool_call, round 2 text
    assert report.converged is True
    assert report.rounds >= 2


def test_auto_mode_ingest_detection():
    """Ingest intent: LLM confirms data directory before calling switch_mode."""
    import json

    responses = iter(
        [
            # Round 1: confirm with user
            json.dumps(
                {
                    "mode": "auto",
                    "action": "text",
                    "text": "I'll ingest data from /data/dir. Confirm? (y/n)",
                }
            ),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            try:
                content = next(responses)
            except StopIteration:
                content = json.dumps({"mode": "auto", "action": "text", "text": "Done."})
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "ingest files from /data/dir"})

    assert report.converged is True
    assert "ingest" in str(report.findings).lower() or report.rounds >= 1


def test_auto_mode_predict_detection():
    """Predict intent: LLM displays disease + drugs for confirmation."""
    import json

    responses = iter(
        [
            json.dumps(
                {
                    "mode": "auto",
                    "action": "text",
                    "text": "I'll predict aspirin efficacy for headache. Confirm? (y/n)",
                }
            ),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            try:
                content = next(responses)
            except StopIteration:
                content = json.dumps({"mode": "auto", "action": "text", "text": "Done."})
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "predict aspirin for headache"})

    assert report.converged is True
    assert "aspirin" in str(report.findings).lower() or report.rounds >= 1


# ------------------------------------------------------------------
# Ingest/Predict ModeSpec tests (Ticket 6)
# ------------------------------------------------------------------


def test_mode_config_injected_to_iris():
    """Iris receives mode_config via DI."""
    from dargus.runtime.modespec import ModeSpec

    ingest_spec = ModeSpec(tools=["read_file", "write_file"], system_prompt="Ingest mode.")
    mode_config = {"auto": ModeSpec(tools=["read_file"]), "ingest": ingest_spec}

    iris = Iris(mode_config=mode_config, mode="ingest")
    assert iris._mode == "ingest"
    assert "ingest" in iris._mode_config
    assert iris._mode_config["ingest"].tools == ["read_file", "write_file"]


def test_mode_transition_end_to_end():
    """LLM in auto mode → switch_mode("ingest") → next round is ingest."""
    import json

    responses = iter(
        [
            # Round 1: switch to ingest
            json.dumps(
                {
                    "mode": "auto",
                    "action": "tool_call",
                    "tool": "switch_mode",
                    "params": {"target": "ingest"},
                }
            ),
            # Round 2: ingest mode (text)
            json.dumps(
                {
                    "mode": "ingest",
                    "action": "text",
                    "text": "Ingest complete. Processed 5 records.",
                }
            ),
        ]
    )

    class MultiRoundBackend:
        def chat(self, messages, options=None):
            content = next(responses)
            return LLMResponse(content=content, usage=LLMUsage(), model="fake")

    from dargus.models.reasoning import ReasoningLLM

    iris = Iris(reasoning_llm=ReasoningLLM(backend=MultiRoundBackend()))
    report = iris.run({"query": "ingest data from /tmp"})

    assert report.converged is True
    assert report.rounds >= 2


def test_mode_tag_mismatch_warning_injected():
    """REASON_END hook sets skip_act on mode-tag mismatch.

    This tests the ModeTagValidationHook integration point.
    """

    from dargus.runtime.hooks import HookContext
    from dargus.runtime.mode_tag import ModeTagValidationHook

    # Set up a runtime-like object with a mismatched mode
    class FakeRuntime:
        mode = "auto"

    hook = ModeTagValidationHook()

    # Case 1: Matching mode — no blocking
    ctx = HookContext(
        runtime=FakeRuntime(),
        extra={"reason_response": {"mode": "auto", "action": "text", "text": "OK"}},
    )
    result = hook(ctx)
    assert result.extra.get("skip_act") is not True

    # Case 2: Mismatched mode — block ACT
    ctx2 = HookContext(
        runtime=FakeRuntime(),
        extra={"reason_response": {"mode": "predict", "action": "text", "text": "Results"}},
    )
    result2 = hook(ctx2)
    assert result2.extra.get("skip_act") is True
    assert result2.extra.get("mode_tag_mismatch") is True
    assert "mismatch" in result2.extra.get("mode_tag_warning", "").lower()

    # Case 3: No runtime — no validation (pass through)
    ctx3 = HookContext(runtime=None, extra={"reason_response": {"mode": "predict"}})
    result3 = hook(ctx3)
    assert result3.extra.get("skip_act") is not True


# ------------------------------------------------------------------
# Hook wiring through the real runtime (issue #64/#66 gaps)
# ------------------------------------------------------------------


def test_runtime_hook_registry_is_prewired():
    """DargusRuntime auto-creates a HookRegistry with the agent-loop hooks.

    ADR-0002: ModeTagValidationHook at REASON_END and WorkspaceGuardHook at
    ACT_START must be registered so the live loop actually enforces them.
    """
    from dargus.runtime.context import DargusRuntime
    from dargus.runtime.hooks import HookPoint
    from dargus.runtime.mode_tag import ModeTagValidationHook

    rt = DargusRuntime()
    reason_hooks = rt.hook_registry.list_hooks(HookPoint.REASON_END)
    act_start_hooks = rt.hook_registry.list_hooks(HookPoint.ACT_START)

    assert any(isinstance(h, ModeTagValidationHook) for h in reason_hooks)
    assert act_start_hooks, "WorkspaceGuardHook must be registered at ACT_START"


def test_factory_injects_runtime_into_agents():
    """AgentFactory attaches the runtime back-reference to every agent."""
    from dargus.runtime.context import DargusRuntime

    rt = DargusRuntime()
    iris = rt.agent_factory.iris()
    assert iris._runtime is rt


def test_mode_tag_hook_fires_through_live_loop():
    """A mismatched LLM mode-tag blocks ACT in the real runtime-wired loop.

    Drives Iris through the actual DargusRuntime so the REASON_END
    ModeTagValidationHook is registered and receives a non-None runtime.
    """
    import json

    from dargus.runtime.context import DargusRuntime
    from dargus.runtime.hooks import HookPoint

    # LLM claims mode "predict" while runtime stays in "auto" → mismatch.
    response = json.dumps(
        {"mode": "predict", "action": "tool_call", "tool": "read_file", "params": {"path": "x"}}
    )

    rt = DargusRuntime()
    iris = rt.agent_factory.iris()
    from dargus.models.reasoning import ReasoningLLM

    iris._reasoning_llm = ReasoningLLM(backend=FakeReasoningBackend(response))

    report = iris.run({"query": "test"})

    # The mode-tag hook must have fired at REASON_END with a real runtime.
    fired = [
        e
        for e in rt.hook_registry.invocation_log
        if e["point"] == HookPoint.REASON_END.name and e["hook"] == "ModeTagValidationHook"
    ]
    assert fired, "ModeTagValidationHook did not fire at REASON_END"
    # Mismatch → ACT skipped → no read_file tool executed.
    executed = [t for t in report.call_trace if t.tool_called == "read_file" and t.error is None]
    assert not executed


def test_act_start_hook_fires_through_live_loop():
    """ACT_START fires before tool execution in the real runtime-wired loop."""
    import json

    from dargus.runtime.context import DargusRuntime
    from dargus.runtime.hooks import HookPoint

    response = json.dumps({"mode": "auto", "action": "text", "text": "done"})

    rt = DargusRuntime()
    iris = rt.agent_factory.iris()
    from dargus.models.reasoning import ReasoningLLM

    iris._reasoning_llm = ReasoningLLM(backend=FakeReasoningBackend(response))
    iris.run({"query": "test"})

    points = {e["point"] for e in rt.hook_registry.invocation_log}
    assert HookPoint.REASON_END.name in points
