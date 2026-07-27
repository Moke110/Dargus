"""Tests for dargus.cli.main — new CLI structure."""

from unittest.mock import patch

import pytest

from dargus.cli.main import main


def test_cli_iris_subcommand_exists(capsys):
    """iris subcommand should exist and require a question."""
    with pytest.raises(SystemExit) as exc:
        main(["iris", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "send a query to Iris" in captured.out or "usage:" in captured.out


def test_cli_config_subcommand_exists(capsys):
    """config subcommand should exist."""
    with pytest.raises(SystemExit) as exc:
        main(["config", "--help"])
    assert exc.value.code == 0


def test_cli_test_subcommand_exists(capsys):
    """test subcommand should exist."""
    with pytest.raises(SystemExit) as exc:
        main(["test", "--help"])
    assert exc.value.code == 0


def test_cli_iris_requires_question(capsys):
    """iris subcommand without question should fail."""
    with pytest.raises(SystemExit) as exc:
        main(["iris"])
    assert exc.value.code != 0


def test_cli_no_subcommand_launches_repl():
    """No subcommand should launch REPL."""
    with patch("dargus.cli.repl.run_repl") as mock_repl:
        main([])
        mock_repl.assert_called_once()


def test_cli_iris_calls_api_ask():
    """iris subcommand should call api.ask."""
    with patch("dargus.api.ask") as mock_ask:
        mock_ask.return_value = "test response"
        main(["iris", "test", "question"])
        mock_ask.assert_called_once_with("test question")


def test_cli_unknown_subcommand_fails(capsys):
    """Unknown subcommand should fail with error."""
    with pytest.raises(SystemExit) as exc:
        main(["unknown"])
    assert exc.value.code != 0
