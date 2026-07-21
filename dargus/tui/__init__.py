"""Dargus TUI — Textual-based interactive terminal interface."""

from __future__ import annotations


def run_app() -> None:
    """Launch the Dargus Textual TUI application."""
    from dargus.tui.app import DargusApp

    app = DargusApp()
    app.run()
