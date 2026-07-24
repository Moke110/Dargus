"""Tests for the bootstrap function."""

from dargus.runtime.bootstrap import bootstrap


class TestBootstrap:
    """Tests for bootstrap() function."""

    def test_bootstrap_with_no_config_creates_minimal_context(self, tmp_path, monkeypatch):
        """When no config file exists, bootstrap creates a minimal RuntimeContext."""
        monkeypatch.chdir(tmp_path)
        ctx = bootstrap()
        assert ctx is not None
        assert ctx.config == {}
        assert ctx.healthy is False  # no models loaded

    def test_bootstrap_with_valid_config(self, tmp_path, monkeypatch):
        """A valid config with env vars creates a healthy RuntimeContext."""
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

        ctx = bootstrap(str(config_path))

        assert ctx.config is not None
        assert ctx.reasoning_llm is not None
        assert ctx.embedding_model is not None
        assert ctx.healthy is True

    def test_bootstrap_handles_missing_api_key(self, tmp_path, monkeypatch):
        """Missing API key env var produces a context that is not healthy."""
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

        # Note: 'anthropic' provider requires model prefix, this may be a partial
        # config. bootstrap should still return a context (even if not fully healthy).
        ctx = bootstrap(str(config_path))

        assert ctx.reasoning_llm is not None
        # embedding_model is None because no embedding config was provided
        assert ctx.embedding_model is None
        # health_check returns False because embedding_model is missing
        assert ctx.healthy is False

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
        custom = tmp_path / "custom_config.yaml"
        custom.write_text(yaml.dump(config))
        monkeypatch.setenv("TEST_KEY", "sk-fake")

        ctx = bootstrap(str(custom))
        assert ctx.healthy is True

    def test_bootstrap_partial_model_config(self, tmp_path, monkeypatch):
        """When model config raises KeyError, bootstrap returns context gracefully."""
        import yaml

        monkeypatch.chdir(tmp_path)

        config = {"models": {"reasoning": {"provider": "anthropic"}}}
        # Missing 'model' field
        config_path = tmp_path / "dargus_config.yaml"
        config_path.write_text(yaml.dump(config))

        ctx = bootstrap(str(config_path))
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.healthy is False
