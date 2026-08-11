"""Standalone smoke: Agents — a real Iris turn settles through the harness.

Pins the agent-loop invariant: running an Iris turn through the real
BaseAgent harness with a scripted stub reasoning LLM produces a converged
AgentReport — the Perceive → Reason → Act loop closes a turn. Offline: no
LLM network call, no real embedding.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_agents.py
"""

from __future__ import annotations

import sys

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


class _StubLLM:
    """Scripted reasoning LLM: returns each queued response in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list = []

    def chat(self, messages) -> object:
        self.calls.append(list(messages))
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = '{"action": "text", "text": "done"}'
        return type("R", (), {"content": content})()


def main() -> int:
    from dargus.iris.commander import Iris

    # A tool_call round then a text reply closes the turn (two PRA rounds).
    iris = Iris(
        reasoning_llm=_StubLLM(
            [
                '{"action": "tool_call", "tool": "read_file", "params": {"path": "/tmp/x"}}',
                '{"action": "text", "text": "concluded"}',
            ]
        )
    )

    report = iris.run({"query": "assess this evidence"})

    # The loop settled: converged with a final text finding.
    assert report.converged is True, f"run did not converge: rounds={report.rounds}"
    assert report.agent_name == "Iris"
    assert report.findings and report.findings[-1] == "concluded"
    assert report.rounds == 2

    # The Session recorded both rounds (user + tool_call + text).
    session = iris._resolve_session({"query": "assess this evidence"})
    roles = [m.role for m in session.messages]
    assert roles == ["user", "assistant", "assistant"], f"unexpected roles {roles}"
    tool_rounds = [m for m in session.messages if m.tool_name is not None]
    assert len(tool_rounds) == 1 and tool_rounds[0].tool_name == "read_file"

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
