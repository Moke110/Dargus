"""Session — the ordered, typed dialogue log that is the single source of
truth for every agent's context (ADR-0005).

An Iris instance is a Session: it owns its dialogue state — metadata plus a
history of Turns, each made of Rounds. ``DargusRuntime`` holds no per-session
state. The in-memory Session is persisted to the workspace archive when it
ends (ADR-0005); summarization/compaction of long Sessions is explicitly
deferred.

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize this Round for the on-disk archive."""
        return {
            "role": self.role,
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "tool_output": self.tool_output,
            "tool_error": self.tool_error,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Round":
        return cls(
            role=data.get("role", ""),
            text=data.get("text", ""),
            tool_name=data.get("tool_name"),
            tool_params=data.get("tool_params") or {},
            tool_output=data.get("tool_output"),
            tool_error=data.get("tool_error"),
            created=data.get("created", _now_iso()),
        )


@dataclass
class Turn:
    """One user prompt plus the final Iris reply it prompts; made of Rounds.

    A Turn is **closed** once the agent's run for its prompt completes (it
    converged or hit MAX_ROUNDS). The coarse ``(prompt, final_reply)``
    summary is what Iris recalls of a closed Turn; its internal Rounds are
    retained for record and replay only. The currently-open Turn (the one in
    flight) projects its full Rounds.
    """

    prompt: str
    rounds: list[Round] = field(default_factory=list)
    closed: bool = False

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize this Turn (prompt + full Rounds) for the archive."""
        return {
            "prompt": self.prompt,
            "rounds": [r.to_dict() for r in self.rounds],
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            prompt=data.get("prompt", ""),
            rounds=[Round.from_dict(r) for r in data.get("rounds", [])],
            closed=bool(data.get("closed", False)),
        )


@dataclass
class SessionMetadata:
    """Persisted metadata for a Session (ADR-0005)."""

    agent: str
    session_id: str = field(default_factory=new_session_id)
    created: str = field(default_factory=_now_iso)
    closed: str | None = None
    workspace_root: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "session_id": self.session_id,
            "created": self.created,
            "closed": self.closed,
            "workspace_root": self.workspace_root,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        return cls(
            agent=data.get("agent", ""),
            session_id=data.get("session_id", new_session_id()),
            created=data.get("created", _now_iso()),
            closed=data.get("closed"),
            workspace_root=data.get("workspace_root", ""),
            turn_count=data.get("turn_count", 0),
        )


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
        """Return the in-flight Turn, creating an open one lazily."""
        if not self.turns or self.turns[-1].closed:
            self.turns.append(Turn(prompt=""))
        return self.turns[-1]

    def add_user(self, text: str) -> Round:
        """Start a new, open Turn with *text* as the user prompt.

        Starting a new Turn closes the previous one (a Turn is one user
        prompt plus its reply). The prompt is Turn state (``turn.prompt``),
        not a logged Round; the returned Round is the projected entry for
        the message stream.
        """
        if self.turns and not self.turns[-1].closed:
            self.turns[-1].closed = True
        turn = Turn(prompt=text)
        self.turns.append(turn)
        self.metadata.turn_count += 1
        return Round.user(turn.prompt)

    def close_current_turn(self) -> None:
        """Close the in-flight Turn (its Rounds project coarse from here on).

        Called by the agent when a run completes — converged or MAX_ROUNDS
        exhausted — so a turn that ends without a text reply is still closed.
        """
        if self.turns:
            self.turns[-1].closed = True

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
        """True for the Turn currently being built by the PRA loop.

        A Turn is in flight until its agent run completes; once closed its
        internal Rounds are retained for record but no longer shown to the
        model. State-based, not reply-based: a run that exhausts MAX_ROUNDS
        with no text reply still closes its Turn.
        """
        return not turn.closed

    def projection(self) -> list[Message]:
        """Project the Session into the ``list[Message]`` ``ReasoningLLM.chat()``
        consumes (ADR-0005 structural rule).

        Closed Turns project as their coarse ``(prompt, final reply)`` summary
        only; the in-flight Turn (the one currently being built) projects its
        full Rounds so multi-round tasks stay coherent. What Iris sees depends
        on a Turn's state — never on how the Session was loaded. A Session
        loaded from the archive has all Turns closed, so a resumed Session
        projects coarse until the next Turn is opened.

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

    # ------------------------------------------------------------------
    # Persistence (ADR-0005 archive)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full Turns→Rounds tree plus metadata.

        This is the on-disk archive shape: the complete record, never a
        projection.
        """
        return {
            "version": 1,
            "metadata": self.metadata.to_dict(),
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            metadata=SessionMetadata.from_dict(data.get("metadata", {})),
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
        )

    def close(self) -> None:
        """Stamp the Session's closed timestamp (once)."""
        if self.metadata.closed is None:
            self.metadata.closed = _now_iso()
