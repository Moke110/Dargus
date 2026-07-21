"""Dargus Textual TUI application with REPL interface."""

from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from dargus import __version__
from dargus.iris.commander import Iris
from dargus.tui.widgets import AgentResponse, HeaderWidget


class DargusApp(App):
    """Interactive TUI for clinical efficacy prediction."""

    CSS = """
    Screen {
        align: center middle;
    }

    #header {
        height: auto;
        margin: 0 0 1 0;
    }

    #input-bar {
        dock: bottom;
        margin: 1 0 0 0;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
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
        yield HeaderWidget(id="header")
        yield AgentResponse(id="response")
        yield Input(placeholder=">  ", id="input-bar")
        yield Static(f"v{__version__}  ·  /help  /quit", id="status-bar")

    def on_mount(self) -> None:
        """Show initial greeting after the UI is ready."""
        response: AgentResponse = self.query_one("#response", AgentResponse)
        if os.environ.get("DARGUS_LLM_API_KEY"):
            response.add_response("Iris: Ready. How can I help with your research?")
        else:
            response.add_response(
                "Iris: No API key configured.\n\n"
                "Set one with:\n"
                "  dargus config set-api-key <provider> <key>\n\n"
                "Examples:\n"
                "  dargus config set-api-key openai sk-...\n"
                "  dargus config set-api-key anthropic sk-ant-..."
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        response: AgentResponse = self.query_one("#response", AgentResponse)

        if value == "/quit":
            self.exit()
            return

        if value == "/help":
            response.add_response(
                "> /help\n"
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
            result = self.iris.process_query(value)
            response.add_response(result)

        self.query_one("#input-bar", Input).value = ""

    def action_quit(self) -> None:
        self.exit()
