"""Tests for Dargus home resolution — the single canonical per-user home (T1).

One resolver returns ``$DARGUS_HOME`` when set, else ``~/.dargus``. Every
path module (config, dbase, sessions, dotenv) consumes it so paths agree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dargus.config.home import (
    dargus_home,
    env_path,
    is_initialized,
    sessions_dir,
    user_config_path,
)


@pytest.fixture
def _clear_home_env(monkeypatch):
    monkeypatch.delenv("DARGUS_HOME", raising=False)


def test_default_home_is_dot_dargus_under_user_home(monkeypatch, tmp_path, _clear_home_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert dargus_home() == tmp_path / ".dargus"


def test_home_env_var_overrides_default(monkeypatch, tmp_path, _clear_home_env):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    assert dargus_home() == home


def test_home_env_var_expands_tilde(monkeypatch, tmp_path, _clear_home_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DARGUS_HOME", "~/.dargus-test")
    assert dargus_home() == tmp_path / ".dargus-test"


def test_relative_home_env_resolves_to_absolute(monkeypatch, tmp_path, _clear_home_env):
    monkeypatch.setenv("DARGUS_HOME", "relative-home")
    assert dargus_home().is_absolute()
    assert dargus_home().name == "relative-home"


def test_derived_paths_resolve_under_home(monkeypatch, tmp_path, _clear_home_env):
    home = tmp_path / "home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    assert user_config_path() == home / "dargus_config.yaml"
    assert env_path() == home / ".env"
    assert sessions_dir() == home / "sessions"


def test_derived_paths_under_default_home(monkeypatch, tmp_path, _clear_home_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".dargus"
    assert user_config_path() == home / "dargus_config.yaml"
    assert env_path() == home / ".env"
    assert sessions_dir() == home / "sessions"


def test_is_initialized_false_without_config(monkeypatch, tmp_path, _clear_home_env):
    home = tmp_path / "home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    assert is_initialized() is False


def test_is_initialized_true_with_config(monkeypatch, tmp_path, _clear_home_env):
    home = tmp_path / "home"
    home.mkdir()
    (home / "dargus_config.yaml").write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setenv("DARGUS_HOME", str(home))
    assert is_initialized() is True


def test_unset_env_var_does_not_crash():
    old = os.environ.pop("DARGUS_HOME", None)
    try:
        assert dargus_home() == Path.home() / ".dargus"
    finally:
        if old is not None:
            os.environ["DARGUS_HOME"] = old
