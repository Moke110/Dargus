"""Dargus Textual TUI application with conversation-style interface."""

from __future__ import annotations

import os

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, RichLog

from dargus import __version__
from dargus.iris.commander import Iris
from dargus.tui._logo import TAGLINE, build_logo

_GREETING = """Hi, I'm Iris, the director agent of Project Dargus.

Dargus is a clinical efficacy prediction system for drug-development
researchers. I coordinate multi-level evidence analysis across molecular,
biomedical, bioinformatics, and clinical domains to predict drug efficacy
with confidence intervals.

You can ask me things like:
  • predict aspirin for migraine
  • what's the evidence for metformin in type 2 diabetes?
  • status"""

_NO_KEY_MESSAGE = """No API key configured. Set one with:
  dargus config set-api-key <provider> <key>"""

_READY_MESSAGE = "How can I help with your research?"


class DargusApp(App):
    """Interactive TUI for clinical efficacy prediction."""

    CSS = """
    #conversation {
        height: 1fr;
    }

    #input-bar {
        dock: bottom;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iris = Iris()

    def compose(self) -> ComposeResult:
        yield RichLog(id="conversation", highlight=True, markup=True)
        yield Input(placeholder=">  ", id="input-bar")

    def on_mount(self) -> None:
        """Write logo and greeting into the conversation log."""
        log = self.query_one("#conversation", RichLog)

        # Logo
        for line in build_logo():
            log.write(line)

        # Tagline
        log.write(Text(TAGLINE, style=Style(color="grey70", italic=True)))

        # Blank line then greeting
        log.write("")
        log.write(Text(_GREETING, style=Style(color="white")))
        log.write(Text(f"\n\nv{__version__}  ·  /help  /quit", style=Style(color="grey50")))
        log.write("")

        # API key status
        if os.environ.get("DARGUS_LLM_API_KEY"):
            log.write(Text(_READY_MESSAGE, style=Style(color="green")))
        else:
            log.write(Text(_NO_KEY_MESSAGE, style=Style(color="yellow")))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        log = self.query_one("#conversation", RichLog)

        if value == "/quit":
            self.exit()
            return

        if value == "/help":
            log.write(Text("> /help", style=Style(color="grey50")))
            log.write(
                "Iris: Available commands:\n"
                "  /help  — show this message\n"
                "  /quit  — exit the TUI\n"
                "\n"
                "Type any natural language query to get started.\n"
                "Examples:\n"
                "  predict aspirin for headache\n"
                "  status\n"
                "  benchmark full stack"
            )
        else:
            log.write(Text(f"> {value}", style=Style(color="grey50")))
            result = self.iris.process_query(value)
            log.write(result)

        self.query_one("#input-bar", Input).value = ""

    def action_quit(self) -> None:
        self.exit()
