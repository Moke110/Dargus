"""Dargus Textual TUI application with REPL interface."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from dargus import __version__
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

    def compose(self) -> ComposeResult:
        yield HeaderWidget(id="header")
        yield AgentResponse(id="response")
        yield Input(placeholder=">  ", id="input-bar")
        yield Static(f"v{__version__}  ·  /help  /quit", id="status-bar")

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
            response.add_response(f"> {value}\n" f"Iris: Processing your request...")

        self.query_one("#input-bar", Input).value = ""

    def action_quit(self) -> None:
        self.exit()
