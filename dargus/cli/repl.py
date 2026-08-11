"""Dargus REPL — interactive clinical efficacy prediction."""

from __future__ import annotations

import logging
import shutil
import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from dargus import __version__
from dargus.cli.ui.logo import DESCRIPTION, build_logo

_HELP = """\
Available commands:
  /help         — show this message
  /new          — start a fresh session (the current one is saved first)
  /resume <id>  — resume an archived session by its id (saves the current one first)
  /quit         — exit (saves the current session)
  /exit         — exit (saves the current session)
  /q            — exit (saves the current session)
  /config       — interactive LLM configuration wizard
  /clear-dbase  — clear all records from the global D-Base
  /test         — run internal test suite

Type any natural language query to get started.

Enter submits your prompt; Shift+Enter / Alt+Enter / Ctrl+J insert a line
break. Up/Down edit within the text first, then step through previous
prompts. Ctrl+C cancels the current input; Ctrl+D on an empty prompt (or
/quit, /exit, /q) exits and saves the session."""

# ── Rich colour palette ───────────────────────────────────────────────────
IRIS_COLOR = "#01F3A2"
ERROR_COLOR = "red"
LITELLM_COLOR = "grey70"
RULE_COLOR = "grey50"

# ── Processing indicator ─────────────────────────────────────────────────


class _ProcessingIndicator:
    """Thread-safe cycling ``[Processing .]`` → ``[Processing ..]`` → ``[Processing ...]``."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _animate(self) -> None:
        dots = 1
        while self._running:
            prefix = Text("[Processing ", style=Style(color="grey70"))
            suffix = Text("]", style=Style(color="grey70"))
            dot_text = "." * dots + " " * (3 - dots)
            indicator = Text(dot_text, style=Style(color=IRIS_COLOR))
            self._console.print(Text.assemble(prefix, indicator, suffix), end="\r")
            dots = (dots % 3) + 1
            time.sleep(0.4)
        # Clear the line after stopping
        self._console.print(" " * 30, end="\r")


# ── LiteLLM log interceptor ───────────────────────────────────────────────


class _LiteLLMHandler(logging.Handler):
    """Route LiteLLM log messages through Rich with a ``[LiteLLM]`` prefix.

    Renders in a neutral grey — informational, not an error (ADR-0007: after
    the local cost-map change, remaining LiteLLM warnings are rare and
    non-fatal).
    """

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._console.print(
            Text.assemble(
                Text("[LiteLLM] ", style=Style(color=LITELLM_COLOR, bold=True)),
                Text(msg, style=Style(color=LITELLM_COLOR)),
            )
        )


def _install_litellm_handler(console: Console) -> None:
    """Replace any plain-text LiteLLM handler with a Rich-coloured one."""
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    for h in list(litellm_logger.handlers):
        litellm_logger.removeHandler(h)
    handler = _LiteLLMHandler(console)
    handler.setLevel(logging.WARNING)
    litellm_logger.addHandler(handler)


def _format_iris_response(text: str) -> Text:
    """Wrap an Iris response with a green ``[Iris]`` label."""
    return Text.assemble(
        Text("[Iris] ", style=Style(color=IRIS_COLOR, bold=True)),
        Text(text),
    )


def _format_error(text: str) -> Text:
    """Wrap a system error with a red ``[DargusRuntime Error]`` label."""
    return Text.assemble(
        Text("[DargusRuntime Error] ", style=Style(color=ERROR_COLOR, bold=True)),
        Text(text),
    )


# ── REPL prompt input seam (ADR-0006) ─────────────────────────────────────


def _app_buffer():
    """The prompt_toolkit buffer of the running application.

    Used by the Ctrl+D condition filters, which run outside a key-binding
    handler where ``event.current_buffer`` is unavailable. During the REPL's
    own prompt the focused buffer is the default one; if a sub-menu (plain
    ``input()``) is active instead, its buffer is the one being evaluated.
    """
    from prompt_toolkit.application import get_app

    return get_app().current_buffer


def _prompt_input(prompt: str) -> str:
    """Read one REPL prompt (the default multiline prompt_toolkit session).

    This module-level seam exists so tests can patch it and drive ``run_repl``
    without a real terminal (prior art: the old ``builtins.input`` patch).
    """
    return _make_prompt_session().prompt(prompt)


#: Shared ``PromptSession`` for the main REPL prompt — multiline, in-memory
#: history only, smart Up/Down, Ctrl+C cancels and Ctrl+D-on-empty exits
#: (ADR-0006). Lazily created so ``run_repl`` stays importable without
#: prompt_toolkit at module load.
_prompt_session: object | None = None


def _make_prompt_session():
    """Construct (once) the multiline ``PromptSession`` for the main prompt.

    Module-level state because prompt_toolkit history accumulates across
    ``prompt()`` calls; recreating the session each prompt would lose recall.
    """
    global _prompt_session
    if _prompt_session is not None:
        return _prompt_session

    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.shortcuts import PromptSession

    history = InMemoryHistory()
    kb = KeyBindings()
    handle = kb.add

    # ── Enter submits; Shift+Enter / Ctrl+Enter insert a line break ──────
    # Enter and the modified variants all parse to ``ControlM`` — the raw
    # ``data`` payload distinguishes them ("\r" is a plain Enter; an escape
    # sequence is a modified one). Alt+Enter arrives as the two-key sequence
    # ``escape`` + ``ControlM`` and is handled separately below.
    @handle(Keys.ControlM)
    def _enter(event) -> None:
        if event.data == "\r":
            # Submit: record the prompt in the in-memory history first
            # (consecutive repeats are collapsed at append time), then exit.
            event.current_buffer.append_to_history()
            event.app.exit(result=event.current_buffer.text)
        else:
            event.current_buffer.insert_text("\n")  # Shift+Enter / Ctrl+Enter

    # Alt+Enter inserts a line break.
    @handle("escape", Keys.ControlM)
    def _alt_enter(event) -> None:
        event.current_buffer.insert_text("\n")

    # Ctrl+J inserts a line break (the other newline chord).
    @handle("c-j")
    def _ctrl_j(event) -> None:
        event.current_buffer.insert_text("\n")

    # ── Smart Up/Down: edit within the text, then step history ───────────
    # Up:  line-up → on the top line, jump to the start of the whole text →
    #      at the start, recall the previous prompt.
    # Down mirrors: line-down → on the last line, jump to the end of the
    # whole text → at the end, step to the next prompt. History recall is
    # prompt_toolkit's own ``history_backward``/``history_forward``, which
    # navigate the buffer's working lines (populated from the in-memory
    # history between prompts; consecutive duplicate submits are collapsed
    # at append time).
    @handle(Keys.Up)
    def _up(event) -> None:
        buffer = event.current_buffer
        if buffer.document.cursor_position_row > 0:
            buffer.cursor_up()
        elif buffer.document.text_before_cursor:
            buffer.cursor_position = 0  # jump to start of the whole text
        else:
            buffer.history_backward()

    @handle(Keys.Down)
    def _down(event) -> None:
        buffer = event.current_buffer
        if buffer.document.cursor_position_row < buffer.document.line_count - 1:
            buffer.cursor_down()
        elif buffer.document.text_after_cursor:
            buffer.cursor_position = len(buffer.text)  # jump to end of text
        else:
            buffer.history_forward()

    # ── Ctrl+C cancels the current input and re-prompts (ADR-0006) ───────
    # The live Session stays live — only /quit, /exit, and Ctrl+D on an
    # empty prompt end the REPL. ``<sigint>`` covers the real-terminal
    # Ctrl+C (SIGINT) path; ``c-c`` covers the ``\x03`` Control-C keypress
    # (e.g. piped input). Both cancel, neither exits.
    @handle(Keys.ControlC)
    @handle("<sigint>")
    def _cancel(event) -> None:
        event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    # ── Ctrl+D: on an empty prompt exits (EOF); with text, delete forward ─
    # Two mutually-exclusive bindings: the empty-buffer one wins only when
    # there is no text, the delete-forward one only when there is.
    from prompt_toolkit.filters import Condition

    @Condition
    def _empty() -> bool:
        doc = _app_buffer().document
        return not doc.text_before_cursor and not doc.text_after_cursor

    @handle(Keys.ControlD, filter=_empty)
    def _eof(event) -> None:
        event.app.exit(exception=EOFError(), style="class:exiting")

    @handle(Keys.ControlD, filter=~_empty)
    def _delete_forward(event) -> None:
        event.current_buffer.delete()

    _prompt_session = PromptSession(
        multiline=True,
        history=history,
        enable_history_search=False,
        key_bindings=kb,
    )
    return _prompt_session


# ── REPL ──────────────────────────────────────────────────────────────────


def _end_live_session(api, console: Console) -> None:
    """Persist-then-end the live Session (runtime exit path)."""
    try:
        api.end_session()
    except Exception:
        console.print(_format_error("Failed to save the current session before exit."))


def run_repl() -> None:
    """Launch the Dargus REPL."""
    from dargus import api
    from dargus.runtime.errors import LLMCallError, NoLLMConfiguredError

    console = Console()

    _install_litellm_handler(console)

    # ── intro block ───────────────────────────────────────────────────────────
    # Logo (boxed) with vertical separator + description — only when terminal is
    # wide enough; text fallback otherwise.
    _LOGO_MIN_WIDTH = 56
    term_w = shutil.get_terminal_size().columns
    if term_w >= _LOGO_MIN_WIDTH:
        logo_lines = build_logo()
        logo_text = Text("\n").join(logo_lines)
        version_text = Text(f"v{__version__}", style=Style(color="grey50"))
        desc_text = Text(DESCRIPTION, style=Style(color="grey70"))
        combined = Text.assemble(logo_text, Text("\n\n"), version_text, Text("\n"), desc_text)
        console.print(Panel(combined, border_style="white", padding=(0, 2)))
    else:
        desc_text = Text(DESCRIPTION, style=Style(color="grey70"))
        console.print(
            Panel(
                Text.assemble(
                    Text("DARGUS (Drug-Argus)", style=Style(color="white", bold=True)),
                    Text("\n"),
                    desc_text,
                ),
                border_style="white",
                padding=(0, 2),
            )
        )

    console.print()

    # ── greeting ──────────────────────────────────────────────────────────────
    if api.has_api_key():
        console.print(
            Text(
                "Hi, I'm Iris, the coordinator agent of Project Dargus. "
                "How can I help with your research?",
                style=Style(color=IRIS_COLOR),
            )
        )
    else:
        console.print(
            Text(
                "No API key configured. Set one with: "
                "dargus config set-api-key <provider> <key>",
                style=Style(color="yellow"),
            )
        )

    console.print()

    # ── Initialize processing indicator ────────────────────────────────────────
    indicator = _ProcessingIndicator(console)

    # ── REPL loop ─────────────────────────────────────────────────────────────
    # Ctrl+C on the prompt cancels the current input and re-prompts; only
    # Ctrl+D on an empty prompt (EOFError) exits — via the persist-then-end
    # path (ADR-0006).
    while True:
        try:
            llm_config = api.get_llm_config()
            model_name = llm_config.get("model", "?")
            console.rule(
                f"[{RULE_COLOR}]{model_name}[/]",
                align="right",
                style=Style(color=RULE_COLOR),
            )
            cmd = _prompt_input("> ").strip()
        except KeyboardInterrupt:
            # Ctrl+C: cancel the half-typed prompt, keep the Session live.
            console.print("^C", style=Style(color="grey50"))
            continue
        except EOFError:
            # Ctrl+D on an empty prompt: persist-then-end, then exit.
            console.print()
            _end_live_session(api, console)
            console.print("Goodbye!", style=Style(color="grey50"))
            break

        if not cmd:
            continue
        if cmd in ("/quit", "/q", "/exit"):
            # Persist-then-end the live Session before leaving — quitting
            # never loses a finished session.
            _end_live_session(api, console)
            console.print("Goodbye!", style=Style(color="grey50"))
            break
        if cmd == "/new":
            try:
                new_id = api.new_session()
            except Exception as exc:
                console.print(_format_error(f"Failed to start a new session: {exc}"))
            else:
                console.print(
                    f"Started a fresh session ({new_id[:8]}…). " "The previous session was saved.",
                    style=Style(color="grey70"),
                )
            console.print()
            continue
        if cmd.startswith("/resume"):
            parts = cmd.split()
            if len(parts) != 2 or not parts[1]:
                console.print(_format_error("Usage: /resume <session-id> — an id is required."))
                console.print()
                continue
            session_id = parts[1]
            try:
                resumed_id = api.resume_session(session_id)
            except FileNotFoundError:
                console.print(
                    _format_error(f"No archived session with id {session_id!r} in this workspace.")
                )
            except Exception as exc:
                console.print(_format_error(f"Failed to resume session: {exc}"))
            else:
                console.print(
                    f"Resumed session {session_id[:8]}… as a fresh continuation "
                    f"({resumed_id[:8]}…).",
                    style=Style(color="grey70"),
                )
            console.print()
            continue
        if cmd in ("/test", "/t"):
            from dargus.cli.commands.test import run_test_menu

            run_test_menu()
            console.print()
            continue
        if cmd == "/help":
            console.print(Panel(_HELP, border_style="white", padding=(0, 2)))
            console.print()
            continue
        if cmd == "/config":
            from dargus.cli.commands.config import run_config_menu

            run_config_menu()
            console.print()
            continue
        if cmd == "/clear-dbase":
            from dargus.cli.commands.config import run_clear_dbase

            run_clear_dbase()
            console.print()
            continue
        # Route to Iris agent via API
        console.print()  # blank line before response
        try:
            indicator.start()
            result = api.ask(cmd)
        except NoLLMConfiguredError as exc:
            indicator.stop()
            console.print(_format_error(str(exc)))
        except LLMCallError as exc:
            indicator.stop()
            console.print(_format_error(str(exc)))
        except Exception as exc:
            indicator.stop()
            # Unexpected error — show traceback context
            console.print(
                _format_error(f"Unexpected error: {exc}\n\n" "This may be a bug. Please report it.")
            )
        else:
            indicator.stop()
            console.print(_format_iris_response(result))
        console.print()
