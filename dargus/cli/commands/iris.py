"""Iris one-shot command — send a query to Iris."""

from __future__ import annotations


def run_iris_query(question: str) -> int:
    """Send a natural language query to Iris and print the response.

    Args:
        question: The natural language query to send to Iris.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from dargus import api
    from dargus.runtime.errors import LLMCallError, NoLLMConfiguredError

    try:
        result = api.ask(question)
    except NoLLMConfiguredError as exc:
        print(f"[DargusRuntime Error] {exc}")
        return 1
    except LLMCallError as exc:
        print(f"[DargusRuntime Error] {exc}")
        return 1

    print(f"[Iris] {result}")
    return 0
