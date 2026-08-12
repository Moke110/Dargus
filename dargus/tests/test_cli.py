"""Tests for dargus.cli.main — CLI dispatch, first-run guards (T5), setup/uninstall."""

from unittest.mock import patch

import pytest

from dargus.cli.main import main


@pytest.fixture(autouse=True)
def dargus_home(tmp_path, monkeypatch):
    """A fresh tmp Dargus home. By default it is *not* initialised so the
    first-run guard (T5) is exercised; tests opt into an initialised home
    with :func:`initialized_home`."""
    home = tmp_path / "dargus_home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    monkeypatch.delenv("DARGUS_CONFIG", raising=False)
    return home


@pytest.fixture
def initialized_home(dargus_home):
    """Mark the tmp home as initialised (setup has written a config)."""
    dargus_home.mkdir(parents=True, exist_ok=True)
    (dargus_home / "dargus_config.yaml").write_text("models: {}\n", encoding="utf-8")
    return dargus_home


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


def test_cli_setup_subcommand_exists(capsys):
    """setup subcommand should exist."""
    with pytest.raises(SystemExit) as exc:
        main(["setup", "--help"])
    assert exc.value.code == 0


def test_cli_uninstall_subcommand_exists(capsys):
    """uninstall subcommand should exist."""
    with pytest.raises(SystemExit) as exc:
        main(["uninstall", "--help"])
    assert exc.value.code == 0


def test_cli_iris_requires_question(capsys):
    """iris subcommand without question should fail."""
    with pytest.raises(SystemExit) as exc:
        main(["iris"])
    assert exc.value.code != 0


def test_cli_no_subcommand_launches_repl(initialized_home):
    """No subcommand should launch REPL."""
    with patch("dargus.cli.repl.run_repl") as mock_repl:
        main([])
        mock_repl.assert_called_once()


def test_cli_iris_calls_api_ask(initialized_home):
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


# ------------------------------------------------------------------
# First-run guards (T5)
# ------------------------------------------------------------------


def test_cli_iris_refuses_on_uninitialised_home(capsys):
    """One-shot iris on an uninitialised home refuses with a setup hint."""
    with patch("dargus.api.ask") as mock_ask:
        code = main(["iris", "hello"])
    assert code == 1
    captured = capsys.readouterr()
    assert "dargus setup" in captured.err
    mock_ask.assert_not_called()


def test_cli_config_refuses_on_uninitialised_home(capsys):

    with patch("dargus.cli.commands.config.run_config_menu", return_value=0) as mock_menu:
        code = main(["config"])
    assert code == 1
    assert "dargus setup" in capsys.readouterr().err
    mock_menu.assert_not_called()


def test_cli_test_refuses_on_uninitialised_home(capsys):

    with patch("dargus.cli.commands.test.run_test_menu", return_value=0) as mock_menu:
        code = main(["test"])
    assert code == 1
    assert "dargus setup" in capsys.readouterr().err
    mock_menu.assert_not_called()


def test_cli_iris_runs_after_setup(initialized_home):
    """After setup has run, the same command works normally."""
    with patch("dargus.api.ask", return_value="ok") as mock_ask:
        code = main(["iris", "hello"])
    assert code == 0
    mock_ask.assert_called_once_with("hello")


# ------------------------------------------------------------------
# setup / uninstall dispatch (T4/T6)
# ------------------------------------------------------------------


def test_cli_setup_dispatches_to_wizard(initialized_home):
    with patch("dargus.cli.commands.setup.run_setup_wizard", return_value=0) as mock_wizard:
        code = main(["setup"])
    assert code == 0
    mock_wizard.assert_called_once_with()


def test_cli_uninstall_dispatches(initialized_home):
    with patch("dargus.cli.commands.uninstall.run_uninstall", return_value=0) as mock_uninstall:
        code = main(["uninstall"])
    assert code == 0
    mock_uninstall.assert_called_once_with()
