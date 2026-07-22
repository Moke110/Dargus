from unittest.mock import patch

import pytest

from dargus.cli import main


def test_cli_train_subcommand_exists(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--datadir" in captured.out


def test_cli_status_subcommand_exists(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["status", "--help"])
    assert exc.value.code == 0


def test_cli_clear_dbase_subcommand_exists(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["clear-dbase", "--help"])
    assert exc.value.code == 0


def test_cli_train_has_reset_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--reset" in captured.out


def test_cli_train_has_disease_kb_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--disease-kb-dir" in captured.out


def test_cli_no_subcommand_starts_tui():
    """Bare 'dargus' (no subcommand) should call run_app() to launch the TUI."""
    with patch("dargus.cli.run_app") as mock_run_app:
        main([])
        mock_run_app.assert_called_once()


def test_cli_status_returns_zero(minimal_dbase):
    """'dargus status' exits 0 on a minimal D-Base."""
    import os

    os.environ["DARGUS_HOME"] = minimal_dbase
    exit_code = main(["status"])
    assert exit_code == 0


def test_cli_clear_dbase_wrong_code_aborts(minimal_dbase, monkeypatch):
    """'dargus clear-dbase' with wrong confirmation code exits 1."""
    import os

    os.environ["DARGUS_HOME"] = minimal_dbase
    monkeypatch.setattr("builtins.input", lambda _: "wrong")
    exit_code = main(["clear-dbase"])
    assert exit_code == 1


def test_cli_clear_dbase_correct_code_clears(minimal_dbase, monkeypatch):
    """'dargus clear-dbase' with correct code exits 0 and prints cleared."""
    import os

    os.environ["DARGUS_HOME"] = minimal_dbase
    monkeypatch.setattr("dargus.cli.secrets.token_hex", lambda _: "abc123def4")
    monkeypatch.setattr("builtins.input", lambda _: "abc123def4")
    exit_code = main(["clear-dbase"])
    assert exit_code == 0


def test_cli_predict_requires_args():
    """'dargus predict' without --drugs or --disease exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        main(["predict"])
    assert exc.value.code != 0


def test_cli_bare_invocation_import_error_handled(capsys):
    """When run_app raises ImportError, CLI prints install hint and exits 1."""
    with patch("dargus.cli.run_app", side_effect=ImportError("No module named 'textual'")):
        exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Cannot launch REPL" in captured.out or "Cannot launch REPL" in captured.err
    assert "pip install" in captured.out or "pip install" in captured.err


def test_cli_bare_invocation_unknown_error_bubbles():
    """Non-ImportError exceptions from run_app are not caught by the guard."""
    with patch("dargus.cli.run_app", side_effect=RuntimeError("unexpected")):
        with pytest.raises(RuntimeError, match="unexpected"):
            main([])


def test_cli_config_subcommand_exists(capsys):
    """'dargus config --help' exists and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["config", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "set-api-key" in captured.out


def test_cli_config_set_api_key_writes_env(tmp_path, monkeypatch):
    """'dargus config set-api-key' writes DARGUS_LLM_API_KEY to .env."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("dargus._env.Path.cwd", lambda: tmp_path)

    exit_code = main(["config", "set-api-key", "openai", "sk-test123"])
    assert exit_code == 0
    content = env_file.read_text()
    assert "DARGUS_LLM_API_KEY=sk-test123" in content


def test_cli_config_show_displays_config(capsys, monkeypatch):
    """'dargus config show' displays LLM config without revealing key."""
    monkeypatch.setenv("DARGUS_LLM_API_KEY", "sk-secret-key")
    exit_code = main(["config", "show"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "********" in captured.out
    assert "sk-secret-key" not in captured.out
    assert "openai" in captured.out or "deepseek" in captured.out


def test_cli_config_show_no_key_warns(capsys, monkeypatch):
    """'dargus config show' with no key set shows guidance."""
    monkeypatch.delenv("DARGUS_LLM_API_KEY", raising=False)
    monkeypatch.setattr("dargus.cli.load_dotenv", lambda *a, **kw: None)
    exit_code = main(["config", "show"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "not set" in captured.out
    assert "set-api-key" in captured.out
