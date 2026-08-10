"""Conversation — the ordered, typed message log that is the single source
of truth for every agent's context (ADR-0003).

This is the lean, opencode-aligned model (map #76, tickets #77/#79):
``ConvMessage`` is one typed entry in the log; ``Conversation`` is the
aggregate owning the ordered messages plus session/parent linkage. The
log is **in-memory and single-process only** — durable persistence and
compaction are explicitly deferred by ADR-0003.

Concepts that sit *alongside* this model, never inside it:
D-Base (orthogonal domain store),
AgentReport/CallTrace (run-level artifacts derived *from* the log),
Hooks (observers that may inject system/synthetic Messages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dargus.models.reasoning import Message

#: The four message roles the log recognises (ticket #79).
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_SYNTHETIC = "synthetic"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCall:
    """The single Tool a round invoked (ticket #77: one tool per round)."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of that Tool — a result dict or an error string."""

    output: Any = None
    error: str | None = None


@dataclass
class ConvMessage:
    """One typed entry in a Conversation log.

    An assistant Message records **at most one** ``tool_call``/``tool_result``
    pair, mirroring one-tool-per-round execution (ticket #77).
    """

    role: str
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    created: str = field(default_factory=_now_iso)

    @classmethod
    def user(cls, text: str) -> "ConvMessage":
        return cls(role=ROLE_USER, text=text)

    @classmethod
    def assistant(cls, text: str = "") -> "ConvMessage":
        return cls(role=ROLE_ASSISTANT, text=text)

    @classmethod
    def system(cls, text: str) -> "ConvMessage":
        return cls(role=ROLE_SYSTEM, text=text)

    @classmethod
    def synthetic(cls, text: str) -> "ConvMessage":
        return cls(role=ROLE_SYNTHETIC, text=text)

    def as_llm_message(self) -> Message:
        """Project this entry into the role/content ``Message`` the
        ``ReasoningLLM.chat()`` interface consumes.

        A tool-carrying assistant message renders the tool name, params, and
        result (or error) as its content — the model sees the whole round.
        """
        if self.role == ROLE_ASSISTANT and self.tool_call is not None:
            name = self.tool_call.name
            params = self.tool_call.params
            content = f"[tool_call] {name} params={params}"
            if self.tool_result is not None:
                if self.tool_result.error is not None:
                    content += f"\n[tool_error] {self.tool_result.error}"
                else:
                    content += f"\n[tool_result] {self.tool_result.output}"
            return Message(role=ROLE_ASSISTANT, content=content)

        return Message(role=self.role, content=self.text)


@dataclass
class Conversation:
    """The aggregate owning an agent's typed, ordered message log.

    Fields:
        session_id: Identifies the owning session (agent-scoped key).
        parent_id: Optional — links a child Conversation to its parent's
            (opencode ``Session.info.parentID`` analogue).
        agent: The agent name this log belongs to.
        messages: The ordered log of :class:`ConvMessage`.
    """

    session_id: str
    agent: str
    parent_id: str | None = None
    messages: list[ConvMessage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Append API
    # ------------------------------------------------------------------

    def add(self, message: ConvMessage) -> ConvMessage:
        """Append a message to the log and return it."""
        self.messages.append(message)
        return message

    def add_user(self, text: str) -> ConvMessage:
        return self.add(ConvMessage.user(text))

    def add_assistant(self, text: str = "") -> ConvMessage:
        return self.add(ConvMessage.assistant(text))

    def add_system(self, text: str) -> ConvMessage:
        return self.add(ConvMessage.system(text))

    def add_synthetic(self, text: str) -> ConvMessage:
        return self.add(ConvMessage.synthetic(text))

    def add_tool(
        self,
        call: ToolCall,
        result: ToolResult,
    ) -> ConvMessage:
        """Append an assistant Message carrying one tool call + result."""
        return self.add(
            ConvMessage(
                role=ROLE_ASSISTANT,
                tool_call=call,
                tool_result=result,
            )
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def to_llm_messages(self) -> list[Message]:
        """Project the log into the ``list[Message]`` ``ReasoningLLM.chat()``
        consumes, replacing the JSON dump of ``history`` + ``act_cache`` that
        used to be built in the agent loop (base.py:422).

        System-prompt framing stays the responsibility of the agent's
        perceive step; this projects the dialogue + tool content.
        """
        return [m.as_llm_message() for m in self.messages]

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def last(self) -> ConvMessage | None:
        return self.messages[-1] if self.messages else None

    def __len__(self) -> int:
        return len(self.messages)
