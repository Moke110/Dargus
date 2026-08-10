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
  /config       — interactive LLM configuration wizard
  /clear-dbase  — clear all records from the global D-Base
  /test         — run internal test suite

Type any natural language query to get started."""

# ── Rich colour palette ───────────────────────────────────────────────────
IRIS_COLOR = "#01F3A2"
ERROR_COLOR = "red"
LITELLM_COLOR = "yellow"
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
    """Route LiteLLM log messages through Rich with a ``[LiteLLM]`` prefix."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._console.print(
            Text.assemble(
                Text("[LiteLLM] ", style=Style(color=LITELLM_COLOR, bold=True)),
                Text(msg),
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
    while True:
        try:
            llm_config = api.get_llm_config()
            model_name = llm_config.get("model", "?")
            console.rule(
                f"[{RULE_COLOR}]{model_name}[/]",
                align="right",
                style=Style(color=RULE_COLOR),
            )
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C / Ctrl-D on the prompt: persist-then-end, then exit.
            console.print()
            _end_live_session(api, console)
            console.print("Goodbye.", style=Style(color="grey50"))
            break

        if not cmd:
            continue
        if cmd in ("/quit", "/q", "/exit"):
            # Persist-then-end the live Session before leaving — quitting
            # never loses a finished session.
            _end_live_session(api, console)
            console.print("Goodbye.", style=Style(color="grey50"))
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
