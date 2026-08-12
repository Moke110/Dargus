"""Tests for dargus.api.setup — the setup wizard API seam (T4, #120).

Drives ``api.setup()`` (never the interactive prompt): a home containing a
valid default config, an optional ``.env``, a D-Base directory structure, and
legacy-session migration that dedupes by ``session_id`` and never overwrites.
Re-running setup on an already-initialised home is safe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import dargus.api as api


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    h = tmp_path / "dargus_home"
    monkeypatch.setenv("DARGUS_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
    (tmp_path / "userhome").mkdir(exist_ok=True)
    return h


class TestSetupHome:
    def test_setup_creates_default_config(self, home: Path):
        result = api.setup(home=home)
        assert (home / "dargus_config.yaml").exists()
        assert result["config"] == str(home / "dargus_config.yaml")
        import yaml

        cfg = yaml.safe_load((home / "dargus_config.yaml").read_text(encoding="utf-8"))
        assert cfg["models"]["reasoning"]["model"]  # valid default config

    def test_setup_writes_api_key_to_home_env(self, home: Path):
        result = api.setup(home=home, api_key="sk-test")
        env = home / ".env"
        assert env.exists()
        assert "DARGUS_LLM_API_KEY=sk-test" in env.read_text(encoding="utf-8")
        assert result["env"] == str(env)

    def test_setup_skips_env_when_no_key(self, home: Path):
        api.setup(home=home)
        assert not (home / ".env").exists()

    def test_setup_creates_dbase_structure(self, home: Path):
        result = api.setup(home=home)
        assert result["dbase"] == str(home / "dbase")
        for sub in ("data", "views", "sidecars"):
            assert (home / "dbase" / sub).is_dir(), f"missing dbase/{sub}"

    def test_setup_never_overwrites_existing_config(self, home: Path):
        home.mkdir(parents=True, exist_ok=True)
        custom = "models:\n  reasoning:\n    provider: anthropic\n    model: custom\n"
        (home / "dargus_config.yaml").write_text(custom, encoding="utf-8")

        api.setup(home=home)
        assert (home / "dargus_config.yaml").read_text(encoding="utf-8") == custom

    def test_setup_re_run_is_safe(self, home: Path):
        first = api.setup(home=home, api_key="sk-1")
        second = api.setup(home=home, api_key="sk-2")
        # Config untouched; key updated in place; no duplicate lines.
        env_text = (home / ".env").read_text(encoding="utf-8")
        assert env_text.count("DARGUS_LLM_API_KEY") == 1
        assert "DARGUS_LLM_API_KEY=sk-2" in env_text
        assert first["config"] == second["config"]


class TestSetupMigration:
    def test_setup_migrates_legacy_workspace_sessions(
        self, home: Path, tmp_path: Path, monkeypatch
    ):
        from dargus.models.session import Session, SessionMetadata
        from dargus.sessions.store import legacy_archive_dir

        # Seed a legacy per-workspace archive in the current cwd.
        workspace = tmp_path / "workspace"
        (legacy_archive_dir(workspace)).mkdir(parents=True, exist_ok=True)
        session = Session(SessionMetadata(agent="Iris", session_id="legacy-1"))
        session.add_user("q1")
        session.add_assistant("a1")
        (legacy_archive_dir(workspace) / "legacy-1.json").write_text(
            __import__("json").dumps(session.to_dict()), encoding="utf-8"
        )
        monkeypatch.chdir(workspace)

        result = api.setup(home=home)
        assert result["migrated"] == 1
        assert (home / "sessions" / "legacy-1.json").exists()

    def test_setup_migration_dedupes_never_overwrites(
        self, home: Path, tmp_path: Path, monkeypatch
    ):
        import json

        from dargus.models.session import Session, SessionMetadata
        from dargus.sessions.store import home_archive_dir, legacy_archive_dir

        workspace = tmp_path / "workspace"
        (home / "sessions").mkdir(parents=True, exist_ok=True)

        # Home already has a "shared" session (newer, authoritative).
        home_session = Session(SessionMetadata(agent="Home", session_id="shared"))
        home_session.add_user("q")
        (home_archive_dir() / "shared.json").write_text(
            json.dumps(home_session.to_dict()), encoding="utf-8"
        )

        # Legacy also has "shared" plus a genuinely new one.
        (legacy_archive_dir(workspace)).mkdir(parents=True, exist_ok=True)
        for sid, agent in (("shared", "Legacy"), ("fresh", "Legacy")):
            s = Session(SessionMetadata(agent=agent, session_id=sid))
            s.add_user("q")
            (legacy_archive_dir(workspace) / f"{sid}.json").write_text(
                json.dumps(s.to_dict()), encoding="utf-8"
            )
        monkeypatch.chdir(workspace)

        result = api.setup(home=home)
        assert result["migrated"] == 1  # only "fresh" is new
        assert (home / "sessions" / "fresh.json").exists()

        from dargus.sessions.store import SessionStore

        loaded = SessionStore(workspace).read("shared")
        assert loaded.metadata.agent == "Home"  # home copy never overwritten


class TestSetupDefaultHome:
    def test_setup_uses_dargus_home_by_default(self, tmp_path: Path, monkeypatch):
        default_home = tmp_path / "default-home"
        monkeypatch.setenv("DARGUS_HOME", str(default_home))
        monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
        (tmp_path / "userhome").mkdir(exist_ok=True)
        result = api.setup()
        assert result["home"] == str(default_home)
        assert (default_home / "dargus_config.yaml").exists()

    def test_custom_home_becomes_canonical_for_first_run_guards(self, tmp_path: Path, monkeypatch):
        """A home chosen at setup survives: the first-run guards then pass
        and uninstall reports the chosen home (Spec US2/US11/US14)."""
        monkeypatch.delenv("DARGUS_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
        (tmp_path / "userhome").mkdir(exist_ok=True)

        custom = tmp_path / "custom-home"
        api.setup(home=custom)

        from dargus.config.home import dargus_home

        assert dargus_home() == custom
        assert api.is_home_initialized() is True  # guards now pass

        with (
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch(
                "subprocess.run",
                return_value=type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
            ),
        ):
            result = api.uninstall()
        assert result["home"] == str(custom)  # US14: reports where data remains

    def test_env_home_override_wins_over_recorded_home(self, tmp_path: Path, monkeypatch):
        """DARGUS_HOME always beats a previously recorded home (T1)."""
        monkeypatch.delenv("DARGUS_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
        (tmp_path / "userhome").mkdir(exist_ok=True)
        custom = tmp_path / "custom-home"
        api.setup(home=custom)

        override = tmp_path / "override-home"
        monkeypatch.setenv("DARGUS_HOME", str(override))

        from dargus.config.home import dargus_home

        assert dargus_home() == override
