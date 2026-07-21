"""Dargus Rich REPL interface."""

from __future__ import annotations


def run_app() -> None:
    """Launch the Dargus REPL. (Deprecated name — kept for compat.)"""
    from dargus.repl import run_repl

    run_repl()
