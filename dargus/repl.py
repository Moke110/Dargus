"""Dargus Rich REPL — interactive clinical efficacy prediction."""

from __future__ import annotations

import logging
import os
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from dargus import __version__
from dargus.iris.commander import Iris
from dargus.tui._logo import TAGLINE, build_logo

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


def _get_current_model() -> str:
    """Read the current LLM model name from dargus_config.yaml."""
    from pathlib import Path

    import yaml

    config_path = Path(__file__).resolve().parent / "config" / "dargus_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("llm", {}).get("model", "?") or "?"


def run_repl() -> None:
    """Launch the Dargus Rich REPL."""
    console = Console()
    iris = Iris()

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
    if os.environ.get("DARGUS_LLM_API_KEY"):
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
            model_name = _get_current_model()
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
            from dargus.cli import _run_test_suite

            _run_test_suite()
            console.print()
            continue
        if cmd == "/help":
            console.print(Panel(_HELP, border_style="white", padding=(0, 2)))
            console.print()
            continue
        if cmd == "/model":
            from dargus.cli import _run_model_wizard

            _run_model_wizard()
            console.print()
            continue
        if cmd == "/clear-dbase":
            from dargus.cli import _clear_dbase

            _clear_dbase()
            console.print()
            continue
        # Route to Iris agent
        console.print()  # blank line before response
        result = iris.process_query(cmd)
        console.print(result)
        console.print()
