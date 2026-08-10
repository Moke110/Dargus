# Plan — Unify Session/Conversation (ADR-0005)

Code plan for the model decided in the grilling session and recorded in
`docs/adr/0005-iris-instance-is-the-session.md` + `CONTEXT.md`. **No code is
changed by this document.** It is the implementation sequence.

## Goal

Collapse `Conversation` into `Session`. An Iris instance owns its live Session
(metadata + Turns⊃Rounds). `DargusRuntime` becomes a pure resource pool holding
one live Iris. Sessions persist to `{workspace}/.dargus/sessions` on Iris end
(runtime exit or swap via `/new` · `/resume`).

## Grain rename (mechanical, do first)

| Today | Becomes |
|---|---|
| `Conversation` (`models/conversation.py`) | `Session` |
| `ConvMessage` | `Round` |
| `ToolCall` / `ToolResult` | folded into `Round` |
| `session_id` + `parent_id` fields | `Session.metadata` |
| `Conversation.to_llm_messages()` | `Session.projection()` |
| `runtime.conversation_store` / `get_conversation()` | **removed** |
| `api._DEFAULT_SESSION_ID` | **removed** |

New first-class type: **`Turn`** — owns its `Round`s + a coarse
`(user_prompt, final_reply)` summary.

## Work items (in order)

1. **Model layer** — `models/session.py`: `Round`, `Turn`, `Session`,
   `SessionMetadata`. Delete `models/conversation.py`. `Session.projection()`
   = coarse prior Turns + current in-flight Turn's full Rounds (structural rule,
   load-path independent).
2. **Ownership move** — Session becomes Iris instance state (`iris._session`),
   not runtime state. `BaseAgent._resolve_conversation()` is replaced: the agent
   reads/appends its own Session directly. Runtime drops `conversation_store`.
3. **Persistence** — on Iris end: serialize Session (full Turns→Rounds tree,
   JSON) to `{workspace}/.dargus/sessions/<session_id>.json`. Triggers: runtime
   exit (`/quit` `/q` `/exit`, Ctrl-C via `atexit`, **not** `__del__`) and swap
   (`/new`, `/resume`). Archive is append-only/immutable.
4. **Swap verb** — one operation, two entry points: `/new` (fresh empty
   Session) and `/resume <id>` (hydrate from archive with a **fresh
   `session_id`**, all loaded Turns projected coarse). Both persist-then-end the
   current live Iris first.
5. **API surface** — `api.ask()` unchanged for dialogue; add
   `api.new_session()` and `api.resume_session(id)`. The cached-runtime +
   cached-Iris reuse (`_RUNTIME_CACHE`, `agent_factory.iris()`) is reworked so
   one runtime holds one live Iris, replaced on swap.
6. **CLI** — `repl.py`: add `/new`, `/resume <id>`, `/exit`; wire existing
   `/quit` `/q` and Ctrl-C/EOF to the persist-then-end path; update `_HELP`.

## Test updates

- `tests/models/test_conversation.py` → `test_session.py` (grain rename,
  projection rule).
- `tests/runtime/test_bootstrap.py`, `test_workspace.py` — drop
  `conversation_store` expectations.
- `tests/agents/test_base.py` — agent owns Session; no runtime store.
- New: persistence round-trip (end → file → resume → fresh id, coarse
  projection), swap semantics (one live Iris; `/new` and `/resume` share the
  verb), `atexit` persist.

## Out of scope (this pass)

- Summarization/compaction of long Sessions (builds later on the archive).
- Multi-live-Iris concurrency (model is strictly one live Iris per runtime).

## Pre-commit gate (CLAUDE.md)

`pytest -q && ruff check dargus && black --check dargus` — on `dev`, never `main`.
