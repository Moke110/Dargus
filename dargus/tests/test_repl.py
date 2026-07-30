"""Tests for dargus.cli.repl REPL."""

import os
from unittest.mock import patch

import pytest

from dargus.cli.repl import _HELP


@pytest.fixture(autouse=True)
def _clear_api_key():
    old = os.environ.pop("DARGUS_LLM_API_KEY", None)
    yield
    if old is not None:
        os.environ["DARGUS_LLM_API_KEY"] = old


def test_greeting_contains_iris():
    """The startup greeting mentions Iris and Dargus."""
    import io

    from dargus.cli.repl import run_repl

    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("builtins.input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "Iris" in output
                assert "Dargus" in output


def test_greeting_contains_key_phrases():
    """The greeting includes 'director and coordinator' and the research offer."""
    import io

    from dargus.cli.repl import run_repl

    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("builtins.input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "coordinator agent" in output
                assert "with your" in output


def test_help_contains_commands():
    """The help text lists available commands."""
    assert "/help" in _HELP
    assert "/quit" in _HELP
    assert "/config" in _HELP
    assert "/clear-dbase" in _HELP


def test_run_repl_quit_command():
    """Typing /quit breaks the REPL loop and prints Goodbye."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye" in output


def test_run_repl_help_command_then_quit():
    """Typing /help prints help, then /quit exits."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["/help", "/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Available commands" in output
            assert "Goodbye" in output


def test_run_repl_empty_input_skipped():
    """Empty input should be skipped without error and not passed to Iris."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["", "", "/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye" in output


def test_run_repl_natural_language_routes_to_iris():
    """Natural language input is routed through iris.process_query."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["predict aspirin for headache", "/quit"]):
        with patch(
            "dargus.api.ask",
            return_value="(mocked) Prediction for aspirin on headache: ...",
        ) as mock_pq:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                mock_pq.assert_called_once_with("predict aspirin for headache")
                assert "(mocked)" in output


def test_run_repl_eof_error_exits():
    """EOFError (Ctrl+D) exits the REPL cleanly."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=EOFError):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye" in output


def test_run_repl_keyboard_interrupt_exits():
    """KeyboardInterrupt (Ctrl+C) exits the REPL cleanly."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye" in output


def test_run_repl_prints_logo_on_startup():
    """The REPL prints the logo on startup."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            # Logo contains distinctive box-drawing character
            assert "█" in output  # █ (full block from logo)


def test_run_repl_prints_description_on_startup():
    """The REPL prints the description on startup."""
    import io

    from dargus.cli.repl import run_repl

    with patch("builtins.input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Clinical drug" in output  # wraps on narrow test terminals


def test_run_repl_api_key_message():
    """Without API key, the REPL shows config guidance; with key, shows ready."""
    import io

    from dargus.cli.repl import run_repl

    # Without API key
    with patch("builtins.input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "No API key configured" in output

    # With API key
    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("builtins.input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "with your" in output
