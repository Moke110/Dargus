"""Tests for the bootstrap function."""

from dargus.runtime.bootstrap import bootstrap
from dargus.runtime.context import DargusRuntime


class TestBootstrap:
    """Tests for bootstrap() function."""

    def test_bootstrap_with_no_config_creates_minimal_runtime(self, tmp_path, monkeypatch):
        """When no config file exists, bootstrap creates a minimal healthy runtime."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DARGUS_CONFIG", str(tmp_path / "nonexistent.yaml"))
        rt = bootstrap()
        assert rt is not None
        assert isinstance(rt, DargusRuntime)
        assert rt.config == {}
        # starts healthy; no models wired, but the flag only flips on failure
        assert rt.healthy is True
        assert rt.reasoning_llm is None

    def test_bootstrap_with_valid_config(self, tmp_path, monkeypatch):
        """A valid config with env vars creates a runtime with both models."""
        import yaml

        monkeypatch.chdir(tmp_path)

        config = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "api_key_env": "TEST_KEY",
                },
                "embedding": {
                    "provider": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2",
                },
            }
        }
        config_path = tmp_path / "dargus_config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("TEST_KEY", "sk-fake")

        rt = bootstrap(str(config_path))

        assert rt.config is not None
        assert rt.reasoning_llm is not None
        assert rt.embedding_model is not None
        assert rt.healthy is True

    def test_bootstrap_handles_missing_api_key(self, tmp_path, monkeypatch):
        """Missing API key env var still produces a runtime (LLM object built lazily)."""
        import yaml

        monkeypatch.chdir(tmp_path)

        config = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                },
            }
        }
        config_path = tmp_path / "dargus_config.yaml"
        config_path.write_text(yaml.dump(config))

        rt = bootstrap(str(config_path))

        assert rt.reasoning_llm is not None
        # embedding_model is None because no embedding config was provided
        assert rt.embedding_model is None

    def test_bootstrap_with_explicit_path(self, tmp_path, monkeypatch):
        """bootstrap with explicit config_path works."""
        import yaml

        monkeypatch.chdir(tmp_path)

        config = {
            "models": {
                "reasoning": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "api_key_env": "TEST_KEY",
                },
                "embedding": {
                    "provider": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2",
                },
            }
        }
        config_path = tmp_path / "custom_config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("TEST_KEY", "sk-fake")

        rt = bootstrap(str(config_path))

        assert rt.config is not None
        assert rt.healthy is True

    def test_bootstrap_wires_factory_and_tool_cache(self, tmp_path, monkeypatch):
        """The runtime auto-wires AgentFactory and a session ToolCache."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DARGUS_CONFIG", str(tmp_path / "nonexistent.yaml"))
        rt = bootstrap()
        assert rt.agent_factory is not None
        assert rt.tool_cache is not None
        rt.shutdown()
