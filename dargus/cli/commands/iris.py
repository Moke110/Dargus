"""Iris one-shot command — send a query to Iris."""

from __future__ import annotations


def run_iris_query(question: str) -> int:
    """Send a natural language query to Iris and print the response.

    Args:
        question: The natural language query to send to Iris.

    Returns:
        Exit code (0 for success).
    """
    from dargus import api

    result = api.ask(question)
    print(result)
    return 0
