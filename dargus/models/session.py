"""Session — the ordered, typed dialogue log that is the single source of
truth for every agent's context (ADR-0005).

An Iris instance is a Session: it owns its dialogue state — metadata plus a
history of Turns, each made of Rounds. ``DargusRuntime`` holds no per-session
state. The log is **in-memory and single-process only** — durable persistence
and compaction are explicitly deferred by ADR-0005.

Grain:
- **Round** — one iteration of the Perceive → Reason → Act loop within a Turn:
  either a single Tool call (with its result) or the reply that closes the Turn.
- **Turn** — one user prompt plus the final Iris reply it prompts; made of
  Rounds.
- **Session** — the ordered sequence of Turns plus metadata.

Concepts that sit *alongside* this model, never inside it:
D-Base (orthogonal domain store),
AgentReport/CallTrace (run-level artifacts derived *from* the log),
Hooks (observers that may inject system/synthetic Rounds).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dargus.models.reasoning import Message

#: The roles the log recognises.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_SYNTHETIC = "synthetic"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    """A fresh, unique Session identity (ADR-0005)."""
    return str(uuid.uuid4())


@dataclass
class Round:
    """One typed entry in a Session — a PRA-loop round within a Turn.

    An assistant Round records **at most one** tool call + result payload,
    folded in directly (there are no separate ``ToolCall``/``ToolResult``
    types; they are a Round's payload).
    """

    role: str
    text: str = ""
    tool_name: str | None = None
    tool_params: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    tool_error: str | None = None
    created: str = field(default_factory=_now_iso)

    @classmethod
    def user(cls, text: str) -> "Round":
        return cls(role=ROLE_USER, text=text)

    @classmethod
    def assistant(cls, text: str = "") -> "Round":
        return cls(role=ROLE_ASSISTANT, text=text)

    @classmethod
    def system(cls, text: str) -> "Round":
        return cls(role=ROLE_SYSTEM, text=text)

    @classmethod
    def synthetic(cls, text: str) -> "Round":
        return cls(role=ROLE_SYNTHETIC, text=text)

    @classmethod
    def tool(
        cls,
        name: str,
        params: dict[str, Any] | None = None,
        output: Any = None,
        error: str | None = None,
    ) -> "Round":
        """An assistant Round carrying one tool call + its outcome."""
        return cls(
            role=ROLE_ASSISTANT,
            tool_name=name,
            tool_params=params or {},
            tool_output=output,
            tool_error=error,
        )

    def as_llm_message(self) -> Message:
        """Project this entry into the role/content ``Message`` the
        ``ReasoningLLM.chat()`` interface consumes.

        A tool-carrying assistant round renders the tool name, params, and
        result (or error) as its content — the model sees the whole round.
        """
        if self.role == ROLE_ASSISTANT and self.tool_name is not None:
            content = f"[tool_call] {self.tool_name} params={self.tool_params}"
            if self.tool_error is not None:
                content += f"\n[tool_error] {self.tool_error}"
            else:
                content += f"\n[tool_result] {self.tool_output}"
            return Message(role=ROLE_ASSISTANT, content=content)

        return Message(role=self.role, content=self.text)


@dataclass
class Turn:
    """One user prompt plus the final Iris reply it prompts; made of Rounds.

    The coarse ``(prompt, final_reply)`` summary is what Iris recalls of a
    closed Turn; its internal Rounds are retained for record and replay only.
    """

    prompt: str
    rounds: list[Round] = field(default_factory=list)

    @property
    def final_reply(self) -> str:
        """The last non-tool assistant text that closes the Turn."""
        for r in reversed(self.rounds):
            if r.role == ROLE_ASSISTANT and r.tool_name is None:
                return r.text
        return ""

    @property
    def summary(self) -> tuple[str, str]:
        """The coarse (user prompt, final reply) exchange."""
        return (self.prompt, self.final_reply)


@dataclass
class SessionMetadata:
    """Persisted metadata for a Session (ADR-0005)."""

    agent: str
    session_id: str = field(default_factory=new_session_id)
    created: str = field(default_factory=_now_iso)
    closed: str | None = None
    workspace_root: str = ""
    turn_count: int = 0


@dataclass
class Session:
    """The aggregate owning a Session's typed, ordered Turns (ADR-0005).

    Fields:
        metadata: Unique session_id, Agent name, timestamps, workspace root,
            turn count.
        turns: The ordered Turns of the Session.
    """

    metadata: SessionMetadata
    turns: list[Turn] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Append API
    # ------------------------------------------------------------------

    def _current_turn(self) -> Turn:
        """Return the in-flight Turn, creating an empty-prompt one lazily."""
        if not self.turns:
            self.turns.append(Turn(prompt=""))
        return self.turns[-1]

    def add_user(self, text: str) -> Round:
        """Start a new Turn with *text* as the user prompt and return its Round."""
        self.turns.append(Turn(prompt=text))
        self.metadata.turn_count += 1
        return Round.user(text)

    def add_assistant(self, text: str = "") -> Round:
        return self._append(Round.assistant(text))

    def add_system(self, text: str) -> Round:
        return self._append(Round.system(text))

    def add_synthetic(self, text: str) -> Round:
        return self._append(Round.synthetic(text))

    def add_tool(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        output: Any = None,
        error: str | None = None,
    ) -> Round:
        """Append an assistant Round carrying one tool call + outcome."""
        return self._append(Round.tool(name, params, output, error))

    def _append(self, round_: Round) -> Round:
        self._current_turn().rounds.append(round_)
        return round_

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[Round]:
        """The flat, ordered entry stream: each Turn's prompt then its Rounds."""
        entries: list[Round] = []
        for turn in self.turns:
            if turn.prompt:
                entries.append(Round.user(turn.prompt))
            entries.extend(turn.rounds)
        return entries

    def _is_in_flight(self, turn: Turn) -> bool:
        """True while the PRA loop is still building *turn*.

        A Turn is in flight until it receives its final reply; once closed
        its internal Rounds are retained for record but no longer shown to
        the model.
        """
        return not bool(turn.final_reply)

    def projection(self) -> list[Message]:
        """Project the Session into the ``list[Message]`` ``ReasoningLLM.chat()``
        consumes (ADR-0005 structural rule).

        Closed Turns project as their coarse ``(prompt, final reply)`` summary
        only; the in-flight Turn (the one currently being built, without a
        final reply) projects its full Rounds so multi-round tasks stay
        coherent. What Iris sees depends on a Turn's state — never on how the
        Session was loaded. A Session loaded from the archive has all Turns
        closed, so a resumed Session projects coarse until the next Turn is
        opened.

        System-prompt framing stays the responsibility of the agent's
        perceive step; this projects the dialogue + tool content.
        """
        projected: list[Message] = []
        for turn in self.turns:
            if self._is_in_flight(turn):
                if turn.prompt:
                    projected.append(Message(role=ROLE_USER, content=turn.prompt))
                projected.extend(r.as_llm_message() for r in turn.rounds)
            else:
                prompt, reply = turn.summary
                if prompt:
                    projected.append(Message(role=ROLE_USER, content=prompt))
                if reply:
                    projected.append(Message(role=ROLE_ASSISTANT, content=reply))
        return projected

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def last(self) -> Round | None:
        entries = self.messages
        return entries[-1] if entries else None

    def __len__(self) -> int:
        return len(self.messages)
