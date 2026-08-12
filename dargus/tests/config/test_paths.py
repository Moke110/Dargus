"""Tests for config file path resolution under the Dargus home (T1)."""

from __future__ import annotations

import pytest


@pytest.fixture
def _clear_config_env(monkeypatch):
    monkeypatch.delenv("DARGUS_CONFIG", raising=False)


def test_config_path_uses_home_override_when_present(monkeypatch, tmp_path, _clear_config_env):
    home = tmp_path / "home"
    home.mkdir()
    (home / "dargus_config.yaml").write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setenv("DARGUS_HOME", str(home))

    from dargus.config.paths import get_config_path

    assert get_config_path() == home / "dargus_config.yaml"


def test_config_path_returns_packaged_default_without_home_config(monkeypatch, tmp_path):
    """No DARGUS_CONFIG, no home config → the packaged default wins."""
    from dargus.config.paths import get_config_path

    monkeypatch.delenv("DARGUS_CONFIG", raising=False)
    monkeypatch.setenv("DARGUS_HOME", str(tmp_path / "empty-home"))
    assert get_config_path().name == "dargus_config.yaml"


def test_config_path_honours_dargus_config_env(monkeypatch, tmp_path):
    from dargus.config.paths import get_config_path

    custom = tmp_path / "custom.yaml"
    custom.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setenv("DARGUS_CONFIG", str(custom))
    assert get_config_path() == custom


def test_dbase_home_resolution_moves_with_dargus_home(monkeypatch, tmp_path):
    from dargus.dbase.paths import dbase_root, default_dargus_home

    home = tmp_path / "home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    assert default_dargus_home() == home
    assert dbase_root() == home / "dbase"
