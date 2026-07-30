"""Dargus command-line interface — argument parsing and dispatch."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dargus`` CLI."""
    from dargus import api

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    api.init()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="dargus",
        description="Dargus — clinical efficacy prediction system",
    )
    subparsers = parser.add_subparsers(dest="command")

    # iris command — send query to Iris
    iris_parser = subparsers.add_parser("iris", help="send a query to Iris")
    iris_parser.add_argument("question", nargs="+", help="natural language query")

    # config command — configuration menu
    subparsers.add_parser("config", help="configuration menu")

    # test command — test menu
    subparsers.add_parser("test", help="test menu")

    args = parser.parse_args(argv)

    # Dispatch to commands
    if args.command == "iris":
        from dargus.cli.commands.iris import run_iris_query

        question = " ".join(args.question)
        return run_iris_query(question)

    if args.command == "config":
        from dargus.cli.commands.config import run_config_menu

        return run_config_menu()

    if args.command == "test":
        from dargus.cli.commands.test import run_test_menu

        return run_test_menu()

    # No command — launch REPL
    from dargus.cli.repl import run_repl

    try:
        run_repl()
    except ImportError as exc:
        print(f"Error: Cannot launch REPL — missing dependency: {exc}", file=sys.stderr)
        print("Run: pip install -e .[dev]", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
