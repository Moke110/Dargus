"""Dargus REPL — interactive clinical efficacy prediction."""

from __future__ import annotations

import logging
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from dargus import __version__
from dargus.cli.ui.logo import DESCRIPTION, build_logo

_HELP = """\
Available commands:
  /help         — show this message
  /quit         — exit
  /config       — interactive LLM configuration wizard
  /clear-dbase  — clear all records from the global D-Base
  /test         — run internal test suite

Type any natural language query to get started."""

# ── Rich colour palette ───────────────────────────────────────────────────
IRIS_COLOR = "#01F3A2"
ERROR_COLOR = "red"
LITELLM_COLOR = "yellow"
RULE_COLOR = "grey50"

# ── LiteLLM log interceptor ───────────────────────────────────────────────


class _LiteLLMHandler(logging.Handler):
    """Route LiteLLM log messages through Rich with a ``[LiteLLM]`` prefix."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._console.print(Text(f"[LiteLLM] {msg}", style=Style(color=LITELLM_COLOR)))


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
        Text(text, style=Style(color=IRIS_COLOR)),
    )


def _format_error(text: str) -> Text:
    """Wrap a system error with a red ``[DargusRuntime Error]`` label."""
    return Text.assemble(
        Text("[DargusRuntime Error] ", style=Style(color=ERROR_COLOR, bold=True)),
        Text(text, style=Style(color=ERROR_COLOR)),
    )


# ── REPL ──────────────────────────────────────────────────────────────────


def run_repl() -> None:
    """Launch the Dargus REPL."""
    from dargus import api
    from dargus.iris.commander import LLMCallError, NoLLMConfiguredError

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
                style=Style(color="green"),
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
            console.print()
            console.print("Goodbye.", style=Style(color="grey50"))
            break

        if not cmd:
            continue
        if cmd in ("/quit", "/q"):
            console.print("Goodbye.", style=Style(color="grey50"))
            break
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
            result = api.ask(cmd)
        except NoLLMConfiguredError as exc:
            console.print(_format_error(str(exc)))
        except LLMCallError as exc:
            console.print(_format_error(str(exc)))
        except Exception as exc:
            # Unexpected error — show traceback context
            console.print(
                _format_error(f"Unexpected error: {exc}\n\n" "This may be a bug. Please report it.")
            )
        else:
            console.print(_format_iris_response(result))
        console.print()
