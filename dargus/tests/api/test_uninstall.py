"""Tests for dargus.api.uninstall — the uninstall API seam (T6, #118).

``dargus uninstall`` removes the program (uv tool uninstall path) while
preserving the Dargus home data, and reports where that data remains. It
never deletes user data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import dargus.api as api


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    h = tmp_path / "dargus_home"
    h.mkdir(parents=True, exist_ok=True)
    (h / "dargus_config.yaml").write_text("models: {}\n", encoding="utf-8")
    (h / ".env").write_text("DARGUS_LLM_API_KEY=sk-x\n", encoding="utf-8")
    (h / "dbase").mkdir()
    (h / "sessions").mkdir()
    monkeypatch.setenv("DARGUS_HOME", str(h))
    return h


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestUninstall:
    def test_delegates_to_uv_tool_uninstall(self, home: Path):
        with (
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch(
                "subprocess.run", return_value=_Proc(0, "Uninstalled dargus-cli", "")
            ) as mock_run,
        ):
            result = api.uninstall()

        assert result["uninstalled"] is True
        assert result["command"] == "uv tool uninstall dargus-cli"
        mock_run.assert_called_once_with(
            ["uv", "tool", "uninstall", "dargus-cli"], capture_output=True, text=True
        )

    def test_preserves_home_data(self, home: Path):
        with (
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch("subprocess.run", return_value=_Proc(0, "", "")),
        ):
            api.uninstall()

        assert (home / "dargus_config.yaml").exists()
        assert (home / ".env").exists()
        assert (home / "dbase").is_dir()
        assert (home / "sessions").is_dir()

    def test_reports_where_data_remains(self, home: Path):
        with (
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch("subprocess.run", return_value=_Proc(0, "", "")),
        ):
            result = api.uninstall()

        data = result["data"]
        assert data["home"] == str(home)
        assert data["config"] == str(home / "dargus_config.yaml")
        assert data["env"] == str(home / ".env")
        assert data["dbase"] == str(home / "dbase")
        assert data["sessions"] == str(home / "sessions")

    def test_noop_without_uv(self, home: Path):
        with patch("shutil.which", return_value=None):
            result = api.uninstall()

        assert result["uninstalled"] is False
        assert "uv not found" in result["error"]
        assert (home / "dargus_config.yaml").exists()  # data untouched

    def test_nonzero_uv_reports_failure(self, home: Path):
        with (
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch("subprocess.run", return_value=_Proc(1, "", "package not found")),
        ):
            result = api.uninstall()

        assert result["uninstalled"] is False
        assert result["error"] == "package not found"
