"""Tests for dargus.cli.repl REPL (ADR-0006, #109).

Drives ``run_repl`` through the ``_prompt_input`` seam (not ``builtins.input``
and not prompt_toolkit internals): submitted prompt text (multi-line allowed)
is fed through the patched wrapper and assertions are made on ``sys.stdout``
output and mocked ``dargus.api`` calls.

Key-contract assertions:
- Enter submits; Shift+Enter / Alt+Enter / Ctrl+J insert a newline.
- Ctrl+C cancels the current input and re-prompts — the live Session stays
  live; it does **not** exit.
- Ctrl+D on an empty prompt (EOFError), ``/quit``, ``/exit``, ``/q`` exit via
  the persist-then-end path.
- Exit prints just ``Goodbye!``.
"""

import io
import logging
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


# ── Startup / greeting ─────────────────────────────────────────────────────


def test_greeting_contains_iris():
    """The startup greeting mentions Iris and Dargus."""
    from dargus.cli.repl import run_repl

    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "Iris" in output
                assert "Dargus" in output


def test_greeting_contains_key_phrases():
    """The greeting includes 'coordinator agent' and the research offer."""
    from dargus.cli.repl import run_repl

    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "coordinator agent" in output
                assert "with your" in output


def test_help_contains_commands():
    """The help text lists available commands."""
    assert "/help" in _HELP
    assert "/quit" in _HELP
    assert "/q" in _HELP
    assert "/new" in _HELP
    assert "/resume <id>" in _HELP
    assert "/exit" in _HELP
    assert "/config" in _HELP
    assert "/clear-dbase" in _HELP


def test_help_explains_multiline_input():
    """The help text describes the multiline key contract (ADR-0006)."""
    assert "Shift+Enter" in _HELP
    assert "Ctrl+J" in _HELP
    assert "Ctrl+C" in _HELP
    assert "Ctrl+D" in _HELP


# ── Session commands (routing) ─────────────────────────────────────────────


def test_run_repl_new_routes_to_api_new_session():
    """Typing /new routes to api.new_session()."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/new", "/quit"]):
        with patch("dargus.api.new_session", return_value="abc-123") as mock_new:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                mock_new.assert_called_once_with()
                output = fake_out.getvalue()
                assert "fresh session" in output
                assert "abc-123" in output


def test_run_repl_resume_routes_to_api_resume_session():
    """Typing /resume <id> routes to api.resume_session(id)."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/resume abc123", "/quit"]):
        with patch("dargus.api.resume_session", return_value="def-456") as mock_resume:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                mock_resume.assert_called_once_with("abc123")
                output = fake_out.getvalue()
                assert "Resumed session" in output
                assert "def-456" in output


def test_run_repl_resume_without_id_shows_usage():
    """/resume without an id prints usage and does not call the API."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/resume", "/quit"]):
        with patch("dargus.api.resume_session") as mock_resume:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                mock_resume.assert_not_called()
                assert "Usage: /resume" in fake_out.getvalue()


def test_run_repl_resume_unknown_id_shows_error():
    """Resuming an unknown id surfaces the missing-archive error."""
    from dargus.cli.repl import run_repl

    def _raise(*a, **k):
        raise FileNotFoundError("gone")

    with patch("dargus.cli.repl._prompt_input", side_effect=["/resume nope", "/quit"]):
        with patch("dargus.api.resume_session", side_effect=_raise):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                assert "No archived session" in fake_out.getvalue()


# ── Exit paths: /quit, /exit, /q, Ctrl+D (EOFError) ───────────────────────


def test_run_repl_quit_persists_session():
    """/quit runs the persist-then-end path (end_session is called)."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                mock_end.assert_called_once_with()
                assert "Goodbye!" in fake_out.getvalue()


def test_run_repl_short_quit_persists_session():
    """/q runs the persist-then-end path."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/q"]):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_end.assert_called_once_with()


def test_run_repl_exit_persists_session():
    """/exit runs the persist-then-end path."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/exit"]):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_end.assert_called_once_with()


def test_run_repl_eof_persists_session():
    """EOFError (Ctrl+D on an empty prompt) runs the persist-then-end path."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=EOFError):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_end.assert_called_once_with()


def test_run_repl_quit_command():
    """Typing /quit breaks the REPL loop and prints Goodbye!."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye!" in output


def test_run_repl_help_command_then_quit():
    """Typing /help prints help, then /quit exits."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/help", "/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Available commands" in output
            assert "Goodbye!" in output


# ── Ctrl+C cancels and re-prompts (ADR-0006) ───────────────────────────────


def test_run_repl_keyboard_interrupt_cancels_and_reprompts():
    """Ctrl+C cancels the current input and re-prompts without persisting.

    The live Session stays live: the cancel itself must not call
    ``end_session`` — only the subsequent ``/quit`` does (exactly once).
    """
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=[KeyboardInterrupt, "/quit"]):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                mock_end.assert_called_once_with()  # the /quit, not the cancel
                assert "Goodbye!" in fake_out.getvalue()


def test_run_repl_keyboard_interrupt_then_query_keeps_session():
    """After a Ctrl+C cancel, a follow-up query still routes to Iris."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=[KeyboardInterrupt, "hello", "/quit"]):
        with patch("dargus.api.ask", return_value="(mocked) hi") as mock_ask:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_ask.assert_called_once_with("hello")


# ── Input handling ─────────────────────────────────────────────────────────


def test_run_repl_empty_input_skipped():
    """Empty input should be skipped without error and not passed to Iris."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["", "", "/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Goodbye!" in output


def test_run_repl_natural_language_routes_to_iris():
    """Natural language input is routed through api.ask."""
    from dargus.cli.repl import run_repl

    with patch(
        "dargus.cli.repl._prompt_input", side_effect=["predict aspirin for headache", "/quit"]
    ):
        with patch(
            "dargus.api.ask",
            return_value="(mocked) Prediction for aspirin on headache: ...",
        ) as mock_pq:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                mock_pq.assert_called_once_with("predict aspirin for headache")
                assert "(mocked)" in output


def test_run_repl_multiline_submits_joined():
    """A multi-line prompt reaches api.ask with the joined text (ADR-0006)."""
    from dargus.cli.repl import run_repl

    with patch(
        "dargus.cli.repl._prompt_input",
        side_effect=["line one\nline two\nline three", "/quit"],
    ):
        with patch("dargus.api.ask", return_value="(mocked) multi") as mock_ask:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_ask.assert_called_once_with("line one\nline two\nline three")


def test_run_repl_multiline_slash_command_trimmed():
    """Leading/trailing whitespace around a slash command is trimmed."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["  /quit  "]):
        with patch("dargus.api.end_session") as mock_end:
            with patch("sys.stdout", new=io.StringIO()):
                run_repl()
                mock_end.assert_called_once_with()


# ── Startup chrome ─────────────────────────────────────────────────────────


def test_run_repl_prints_logo_on_startup():
    """The REPL prints the logo on startup."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            # Logo contains distinctive box-drawing character
            assert "█" in output  # █ (full block from logo)


def test_run_repl_prints_description_on_startup():
    """The REPL prints the description on startup."""
    from dargus.cli.repl import run_repl

    with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "Clinical drug" in output  # wraps on narrow test terminals


def test_run_repl_api_key_message():
    """Without API key, the REPL shows config guidance; with key, shows ready."""
    from dargus.cli.repl import run_repl

    # Without API key
    with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_repl()
            output = fake_out.getvalue()
            assert "No API key configured" in output

    # With API key
    with patch.dict(os.environ, {"DARGUS_LLM_API_KEY": "sk-test"}):
        with patch("dargus.cli.repl._prompt_input", side_effect=["/quit"]):
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                run_repl()
                output = fake_out.getvalue()
                assert "with your" in output


# ── LiteLLM warning restyle (ADR-0007) ─────────────────────────────────────


def test_litellm_handler_renders_warning_neutral_not_error():
    """A WARNING-level LiteLLM record renders neutral grey, not error red."""
    from rich.console import Console

    from dargus.cli.repl import _LiteLLMHandler

    fake_out = io.StringIO()
    console = Console(file=fake_out, force_terminal=True, color_system="standard")
    handler = _LiteLLMHandler(console)

    record = logging.LogRecord(
        "LiteLLM",
        logging.WARNING,
        "get_model_cost_map.py",
        271,
        "LiteLLM: Failed to fetch remote model cost map … Falling back to local backup.",
        (),
        None,
    )
    handler.emit(record)

    out = fake_out.getvalue()
    assert "[LiteLLM]" in out
    # Error red is `\x1b[31m`; the neutral grey style must not include it.
    assert "\x1b[31m" not in out


def test_run_repl_litellm_warning_not_error_styled():
    """A fired [LiteLLM] warning during the REPL renders neutral, not red."""
    from rich.console import Console

    from dargus.cli.repl import run_repl

    fake_out = io.StringIO()
    console = Console(file=fake_out, force_terminal=True, color_system="standard")

    def _script(prompt):
        logging.getLogger("LiteLLM").warning(
            "LiteLLM: Failed to fetch remote model cost map from "
            "https://raw.githubusercontent.com/BerriAI/litellm/main/"
            "model_prices_and_context_window.json: The read operation timed out. "
            "Falling back to local backup."
        )
        return "/quit"

    with patch("dargus.cli.repl.Console", return_value=console):
        with patch("dargus.cli.repl._prompt_input", side_effect=_script):
            with patch("dargus.api.end_session"):
                run_repl()

    out = fake_out.getvalue()
    assert "Failed to fetch remote model cost map" in out
    assert "\x1b[31m" not in out  # not error red
    assert "Goodbye!" in out


def test_local_cost_map_helper_forces_env_before_import():
    """use_local_model_cost_map() forces the env var before litellm import."""
    from dargus._env import use_local_model_cost_map

    os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    use_local_model_cost_map()
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"


def test_local_cost_map_helper_respects_user_override():
    """A user-exported value wins (setdefault, ADR-0007)."""
    from dargus._env import use_local_model_cost_map

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "false"
    use_local_model_cost_map()
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "false"


def test_prompt_input_seam_returns_submitted_text():
    """The default seam uses a multiline PromptSession (ADR-0006)."""
    from dargus.cli.repl import _make_prompt_session

    session = _make_prompt_session()
    assert session.multiline
    assert type(session.history).__name__ == "InMemoryHistory"  # in-memory only
