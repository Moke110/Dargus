"""Dargus TUI custom widgets."""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from dargus.tui._logo import STYLE_TAGLINE, TAGLINE, build_logo


class HeaderWidget(Static):
    """Displays the Dargus ASCII logo and tagline, with a dim blue CSS border."""

    DEFAULT_CSS = """
    HeaderWidget {
        height: auto;
        margin: 0 0 1 0;
        content-align: center middle;
        border: solid white;
        padding: 0 2;
    }
    """

    def __init__(self, *args, **kwargs):
        logo_lines = build_logo()
        tagline_text = Text(TAGLINE, STYLE_TAGLINE)
        self._inner_rich = Text.assemble(Text("\n").join(logo_lines), Text("\n\n"), tagline_text)
        super().__init__(self._inner_rich, *args, **kwargs)

    def render(self):
        """Return the logo/header text (unwrapped for direct inspection)."""
        return self._inner_rich


class AgentResponse(VerticalScroll):
    """Scrollable area for Iris agent responses."""

    can_focus = False

    DEFAULT_CSS = """
    AgentResponse {
        height: 1fr;
        border: solid white;
        padding: 0 1;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response = Text(
            "Iris: How can I help with your research?\n"
            "Type your query below or /help for commands.",
            style=Style(color="white"),
        )

    def render(self) -> Text:
        """Return the current response text."""
        return self._response

    def add_response(self, message: str) -> None:
        """Append a message to the response area."""
        current = self._response.plain
        self._response = Text(
            current + "\n\n" + message,
            style=Style(color="white"),
        )
        self.refresh(layout=True)
