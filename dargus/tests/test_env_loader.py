"""Tests for dargus._env — .env loader/writer under the Dargus home (T3).

Secrets live in ``{home}/.env`` with 0600 permissions and load home-first;
``cwd/.env`` remains a backward-compatible fallback when no home ``.env``
exists.
"""

import os
import stat
from pathlib import Path

import pytest

from dargus._env import load_dotenv, write_dotenv


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot os.environ so setdefault-based loads don't leak across tests."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def dargus_home_tmp(tmp_path: Path, monkeypatch):
    home = tmp_path / "dargus_home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DARGUS_HOME", str(home))
    return home


def test_load_dotenv_parses_key_value_pairs(tmp_path: Path):
    """load_dotenv reads KEY=VALUE lines and sets them in os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n")

    # Use unique keys to avoid cross-test pollution
    load_dotenv(env_path=str(env_file))
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_load_dotenv_skips_comments_and_blanks(tmp_path: Path):
    """load_dotenv ignores comment lines and blank lines."""
    env_file = tmp_path / ".env"
    env_file.write_text("# this is a comment\n\nKEY1=val1\n\n# another comment\nKEY2=val2\n")

    load_dotenv(env_path=str(env_file))
    assert os.environ["KEY1"] == "val1"
    assert os.environ["KEY2"] == "val2"
    assert "# this is a comment" not in os.environ


def test_load_dotenv_setdefault_preserves_existing(tmp_path: Path):
    """load_dotenv uses setdefault — existing env vars take precedence."""
    os.environ["EXISTING_VAR"] = "original"
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=overridden\n")

    load_dotenv(env_path=str(env_file))
    assert os.environ["EXISTING_VAR"] == "original"


def test_load_dotenv_none_path_searches_home_then_cwd(tmp_path: Path, monkeypatch):
    """load_dotenv() loads {home}/.env first, then cwd/.env (T3)."""
    home = tmp_path / "dargus_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("HOME_VAR=home-value\n", encoding="utf-8")
    monkeypatch.setenv("DARGUS_HOME", str(home))

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("CWD_VAR=cwd-value\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    load_dotenv()
    assert os.environ["HOME_VAR"] == "home-value"
    assert os.environ["CWD_VAR"] == "cwd-value"


def test_load_dotenv_home_wins_over_cwd(tmp_path: Path, monkeypatch):
    """A key in both files resolves to the home value (home-first, T3)."""
    home = tmp_path / "dargus_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("SHARED=home\n", encoding="utf-8")
    monkeypatch.setenv("DARGUS_HOME", str(home))

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("SHARED=cwd\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    load_dotenv()
    assert os.environ["SHARED"] == "home"


def test_load_dotenv_none_path_cwd_fallback(tmp_path: Path, monkeypatch):
    """A key in cwd/.env still resolves when no home .env exists (T3)."""
    monkeypatch.setenv("DARGUS_HOME", str(tmp_path / "dargus_home"))

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("CWD_VAR=found\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    load_dotenv()
    assert os.environ["CWD_VAR"] == "found"


def test_load_dotenv_file_not_found_noop(tmp_path: Path):
    """load_dotenv with nonexistent path does nothing, no error."""
    load_dotenv(env_path=str(tmp_path / "nonexistent.env"))
    # No exception = pass


# ------------------------------------------------------------------
# write_dotenv → {home}/.env (T3)
# ------------------------------------------------------------------


def test_write_dotenv_defaults_to_home_env(dargus_home_tmp: Path):
    """write_dotenv() writes to {home}/.env by default (T3)."""
    written = write_dotenv("DARGUS_LLM_API_KEY", "sk-123")
    assert written == str(dargus_home_tmp / ".env")
    assert (
        dargus_home_tmp.joinpath(".env")
        .read_text(encoding="utf-8")
        .startswith("DARGUS_LLM_API_KEY=sk-123")
    )


def test_write_dotenv_sets_0600_permissions(dargus_home_tmp: Path):
    write_dotenv("DARGUS_LLM_API_KEY", "sk-secret")
    mode = stat.S_IMODE(dargus_home_tmp.joinpath(".env").stat().st_mode)
    assert mode == 0o600


def test_write_dotenv_creates_home_dir(tmp_path: Path, monkeypatch):
    home = tmp_path / "nested" / "home"
    monkeypatch.setenv("DARGUS_HOME", str(home))
    written = write_dotenv("K", "v")
    assert Path(written).exists()


def test_write_dotenv_explicit_path_still_honoured(tmp_path: Path):
    target = tmp_path / "custom.env"
    written = write_dotenv("K", "v", env_path=str(target))
    assert written == str(target)
    assert target.exists()


def test_write_dotenv_updates_existing_key_in_place(dargus_home_tmp: Path):
    write_dotenv("DARGUS_LLM_API_KEY", "first")
    write_dotenv("DARGUS_LLM_API_KEY", "second")
    content = dargus_home_tmp.joinpath(".env").read_text(encoding="utf-8")
    assert content.count("DARGUS_LLM_API_KEY") == 1
    assert "DARGUS_LLM_API_KEY=second" in content


def test_home_env_path_used_by_api_set_api_key(dargus_home_tmp: Path, monkeypatch):
    """api.set_api_key writes to the home .env (T3)."""
    import dargus.api as api

    env_path = api.set_api_key("deepseek", "sk-key")
    assert env_path == str(dargus_home_tmp / ".env")
    loaded = dargus_home_tmp.joinpath(".env").read_text(encoding="utf-8")
    assert "DARGUS_LLM_API_KEY=sk-key" in loaded
