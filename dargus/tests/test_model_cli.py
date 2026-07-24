"""Tests for dargus model interactive wizard."""

from unittest.mock import patch

import httpx

from dargus.llm import DargusLLM, check_llm_connection


def test_check_llm_connection_ok():
    """check_llm_connection returns ok=True on successful response."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    )
    llm = DargusLLM(
        model="test",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=transport),
    )
    result = check_llm_connection(llm)
    assert result["ok"] is True
    assert result["model"] == "test"
    assert "latency_ms" in result


def test_check_llm_connection_fail():
    """check_llm_connection returns ok=False on HTTP error."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "unauthorized"})
    )
    llm = DargusLLM(
        model="bad",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=transport),
    )
    result = check_llm_connection(llm)
    assert result["ok"] is False
    assert "error" in result


def test_run_model_wizard_help_exists():
    """dargus model --help is a valid CLI subcommand."""
    from dargus.cli import main

    with patch("sys.stdout"):
        with patch("sys.stderr"):
            with patch("dargus.cli.main._run_model_wizard", return_value=0):
                result = main(["model"])
                assert result == 0


def test_run_model_wizard_skip(monkeypatch, capsys):
    """Choosing 'Skip' from menu returns 0 without entering config flow."""
    from dargus.cli.main import _run_model_wizard

    with patch("dargus.cli.main._arrow_menu", return_value=1):
        result = _run_model_wizard()
    assert result == 0
    captured = capsys.readouterr()
    assert "Keeping current configuration" in captured.out


def test_run_model_wizard_enter_config(monkeypatch, capsys):
    """Choosing 'Enter new configuration' then Discard works."""
    from dargus.cli.main import _run_model_wizard

    monkeypatch.setenv("DARGUS_LLM_API_KEY", "sk-test")
    inputs = ["", "", "", "n"]
    with patch("dargus.cli.main._arrow_menu", return_value=0):
        with patch("builtins.input", side_effect=inputs):
            result = _run_model_wizard()
    assert result == 0
    captured = capsys.readouterr()
    assert "Discarded" in captured.out
