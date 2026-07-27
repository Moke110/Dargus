"""Dargus REPL — interactive clinical efficacy prediction."""

from __future__ import annotations

import logging
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from dargus import __version__
from dargus.cli.ui.logo import TAGLINE, build_logo

_GREETING = """\
Hi, I'm Iris, the director agent of Project Dargus.

Dargus is a clinical efficacy prediction system for drug-development
researchers. I coordinate multi-level evidence analysis across molecular,
biomedical, bioinformatics, and clinical domains to predict drug efficacy
with confidence intervals.

You can ask me things like:
  • predict aspirin for migraine
  • what's the evidence for metformin in type 2 diabetes?
  • status"""

_HELP = """\
Available commands:
  /help         — show this message
  /quit         — exit
  /model        — interactive LLM configuration wizard
  /clear-dbase  — clear all records from the global D-Base
  /test         — run internal test suite

Type any natural language query to get started."""


def run_repl() -> None:
    """Launch the Dargus REPL."""
    from dargus import api

    console = Console()

    logging.getLogger("httpx").setLevel(logging.WARNING)

    # ── intro block ───────────────────────────────────────────────────────────────
    # Logo (boxed) — only when terminal is wide enough; text fallback otherwise
    _LOGO_MIN_WIDTH = 56
    term_w = shutil.get_terminal_size().columns
    if term_w >= _LOGO_MIN_WIDTH:
        logo_lines = build_logo()
        logo_text = Text("\n").join(logo_lines)
        tagline_text = Text(TAGLINE, style=Style(color="grey70", italic=True))
        version_text = Text(f"v{__version__}", style=Style(color="grey50"))
        combined = Text.assemble(
            logo_text, Text("\n"), tagline_text, Text("\n\n"), version_text, Text("  ")
        )
        console.print(Panel(combined, border_style="white", padding=(0, 2)))
    else:
        console.print(
            Panel(
                Text("DARGUS (Drug-Argus)", style=Style(color="white", bold=True)),
                border_style="white",
                padding=(0, 2),
            )
        )

    console.print()
    console.print(Panel(_GREETING, border_style="white", padding=(1, 2)))

    console.print()
    console.print(Text(f"v{__version__}  ·  /help  /quit  /model", style=Style(color="grey50")))

    # API key status
    if api.has_api_key():
        console.print(Text("How can I help with your research?", style=Style(color="green")))
    else:
        console.print(
            Text(
                "No API key configured. Set one with: "
                "dargus config set-api-key <provider> <key>",
                style=Style(color="yellow"),
            )
        )

    console.print()

    # ── REPL loop ───────────────────────────────────────────────────────────────
    while True:
        try:
            llm_config = api.get_llm_config()
            model_name = llm_config.get("model", "?")
            console.rule(f"[grey50]{model_name}[/]", align="right", style=Style(color="grey50"))
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
        if cmd == "/model":
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
        result = api.ask(cmd)
        console.print(result)
        console.print()
