"""Tests for the dargus setup wizard — CLI rendering around the API seam (T4).

The wizard is interactive-only; the file operations it drives (config, .env,
D-Base, migration) are covered at the API seam. Here we assert the prompt
wiring: home confirmation, the skippable API-key step, and the summary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dargus.cli.commands.setup import run_setup_wizard


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    h = tmp_path / "dargus_home"
    monkeypatch.setenv("DARGUS_HOME", str(h))
    return h


class TestSetupWizard:
    def test_wizard_honours_entered_home(self, home: Path, monkeypatch, capsys):
        entered = str(home / "my-home")
        monkeypatch.setattr("builtins.input", lambda _: entered if "Dargus home" in _ else "n")

        with patch(
            "dargus.api.setup",
            return_value={
                "home": entered,
                "config": f"{entered}/dargus_config.yaml",
                "env": None,
                "dbase": f"{entered}/dbase",
                "migrated": 0,
            },
        ) as mock_setup:
            code = run_setup_wizard()

        assert code == 0
        mock_setup.assert_called_once_with(home=Path(entered), api_key=None)
        assert entered in capsys.readouterr().out

    def test_wizard_defaults_to_dargus_home(self, home: Path, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")

        with patch(
            "dargus.api.setup",
            return_value={
                "home": str(home),
                "config": str(home / "dargus_config.yaml"),
                "env": None,
                "dbase": str(home / "dbase"),
                "migrated": 0,
            },
        ) as mock_setup:
            run_setup_wizard()

        mock_setup.assert_called_once_with(home=None, api_key=None)

    def test_wizard_skips_api_key_by_default(self, home: Path, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")

        with (
            patch(
                "dargus.api.setup",
                return_value={
                    "home": str(home),
                    "config": str(home / "dargus_config.yaml"),
                    "env": None,
                    "dbase": str(home / "dbase"),
                    "migrated": 0,
                },
            ),
            patch("dargus.api.set_api_key") as mock_key,
        ):
            run_setup_wizard()

        mock_key.assert_not_called()

    def test_wizard_passes_api_key_when_requested(self, home: Path, monkeypatch):
        answers = iter(["", "y", "sk-key"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        with patch(
            "dargus.api.setup",
            return_value={
                "home": str(home),
                "config": str(home / "dargus_config.yaml"),
                "env": str(home / ".env"),
                "dbase": str(home / "dbase"),
                "migrated": 0,
            },
        ) as mock_setup:
            run_setup_wizard()

        mock_setup.assert_called_once_with(home=None, api_key="sk-key")

    def test_wizard_empty_key_is_skipped(self, home: Path, monkeypatch):
        answers = iter(["", "y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        with patch(
            "dargus.api.setup",
            return_value={
                "home": str(home),
                "config": str(home / "dargus_config.yaml"),
                "env": None,
                "dbase": str(home / "dbase"),
                "migrated": 0,
            },
        ) as mock_setup:
            run_setup_wizard()

        mock_setup.assert_called_once_with(home=None, api_key=None)

    def test_wizard_prints_summary(self, home: Path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "")
        with patch(
            "dargus.api.setup",
            return_value={
                "home": str(home),
                "config": str(home / "dargus_config.yaml"),
                "env": None,
                "dbase": str(home / "dbase"),
                "migrated": 2,
            },
        ):
            run_setup_wizard()

        out = capsys.readouterr().out
        assert "Dargus is set up" in out
        assert str(home) in out
        assert "migrated" in out
