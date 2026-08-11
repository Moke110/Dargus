"""Standalone smoke: API — new/resume/end session through the public facade.

Pins the session-lifecycle invariant on the ``dargus.api`` seam: a real
``new_session`` / ``ask`` / ``resume_session`` / ``end_session`` round-trip in
an isolated tmp workspace with a stubbed reasoning LLM. Proves the public
facade wires real persistence (archive on disk) without touching the real
``.dargus/`` or making any LLM network call.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_api.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


class _ScriptedBackend:
    """Reasoning backend returning queued PRA responses, then a default."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def chat(self, messages, options=None):
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = '{"action": "text", "text": "done"}'
        return type("R", (), {"content": content})()


def _archive_ids(workspace: Path) -> list[str]:
    from dargus.sessions.store import SessionStore

    return SessionStore(str(workspace)).list_ids()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()

        # Drive dargus.api with a runtime wired to a tmp workspace + stub LLM.
        import dargus.api as api
        from dargus.models.reasoning import ReasoningLLM
        from dargus.runtime.context import DargusRuntime

        rt = DargusRuntime(config={"workspace_root": str(workspace)})
        rt.reasoning_llm = ReasoningLLM(
            backend=_ScriptedBackend(
                [
                    '{"action": "text", "text": "first reply"}',
                    '{"action": "text", "text": "resumed reply"}',
                ]
            )
        )
        api._RUNTIME_CACHE = rt

        try:
            # 1. Ask through the public facade; the stub LLM replies.
            reply = api.ask("what is aspirin?")
            assert "first reply" in reply, f"unexpected reply {reply!r}"

            live = rt.agent_factory._iris_cache._session
            archived_id = live.metadata.session_id

            # 2. new_session persists-then-starts-fresh; archive gains a file.
            new_id = api.new_session()
            assert new_id != archived_id
            assert archived_id in _archive_ids(workspace)

            # 3. resume brings the archived session back under a fresh identity.
            resumed_id = api.resume_session(archived_id)
            assert resumed_id != archived_id
            resumed = rt.agent_factory._iris_cache._session
            assert resumed.metadata.session_id == resumed_id
            assert len(resumed.turns) == 1  # prior turn loaded coarse

            # 4. The archived original is untouched; end persists the live one.
            assert archived_id in _archive_ids(workspace)
            ended_id = api.end_session()
            assert ended_id is not None
            assert ended_id in _archive_ids(workspace)
        finally:
            api._RUNTIME_CACHE = None  # do not let the atexit net fire on a stale runtime

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
