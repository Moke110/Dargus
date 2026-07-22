"""Tests for dargus._env — .env file loader."""

import os
from pathlib import Path

from dargus._env import load_dotenv


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


def test_load_dotenv_none_path_searches_cwd(tmp_path: Path, monkeypatch):
    """load_dotenv with path=None searches cwd for .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("CWD_VAR=found\n")
    monkeypatch.chdir(tmp_path)

    load_dotenv()
    assert os.environ["CWD_VAR"] == "found"


def test_load_dotenv_file_not_found_noop(tmp_path: Path):
    """load_dotenv with nonexistent path does nothing, no error."""
    load_dotenv(env_path=str(tmp_path / "nonexistent.env"))
    # No exception = pass
