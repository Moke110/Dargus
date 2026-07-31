"""Tests for dargus.cli.commands.config — model wizard."""

from unittest.mock import patch

from dargus.cli.commands.config import _run_model_wizard, run_config_menu


def test_run_config_menu_help_exists(capsys):
    """Config menu should show options."""
    with patch("builtins.input", return_value="5"):
        run_config_menu()
    captured = capsys.readouterr()
    assert "Dargus Configuration" in captured.out
    assert "Show current LLM configuration" in captured.out


def test_run_model_wizard_skip(monkeypatch, capsys):
    """Model wizard with empty inputs should keep current config."""
    inputs = iter(["", "", "", "", "n"])  # provider, base_url, model, api_key, save_confirm
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("dargus.api.test_llm_connection") as mock_test:
        mock_test.return_value = {"ok": False, "error": "test", "model": "", "latency_ms": 0}
        _run_model_wizard()

    captured = capsys.readouterr()
    assert "Configure LLM connection" in captured.out
    assert "Select LLM provider" in captured.out


def test_run_model_wizard_enter_config(monkeypatch, capsys):
    """Model wizard with new config should save."""
    inputs = iter(["2", "http://test.com/v1", "test-model", "test-key", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("dargus.api.test_llm_connection") as mock_test:
        mock_test.return_value = {
            "ok": True,
            "model": "test-model",
            "latency_ms": 100,
            "error": "",
        }
        with patch("dargus.api.save_llm_config") as mock_save:
            with patch("dargus.api.set_api_key") as mock_set_key:
                _run_model_wizard()
                mock_save.assert_called_once_with("test-model", "http://test.com/v1", "openai")
                mock_set_key.assert_called_once_with("default", "test-key")

    captured = capsys.readouterr()
    assert "Configuration saved" in captured.out
