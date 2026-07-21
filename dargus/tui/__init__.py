"""Dargus TUI — Textual-based interactive terminal interface."""

from __future__ import annotations


def run_app() -> None:
    """Launch the Dargus Textual TUI application."""
    try:
        from dargus.tui.app import DargusApp
    except ImportError as exc:
        raise ImportError(f"textual (TUI framework) — {exc}") from exc

    app = DargusApp()
    app.run()
