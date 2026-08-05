"""Tests for DargusRuntime dataclass, health flag, health_check, and mode system."""

import pytest

from dargus.runtime.context import (
    DargusRuntime,
    _mode_config_from_config,
    health_check,
)
from dargus.runtime.modespec import ModeSpec, default_mode_config
from dargus.tools.cache import ToolCache


class TestDargusRuntime:
    """Tests for DargusRuntime creation and defaults."""

    def test_default_construction(self):
        rt = DargusRuntime()
        assert rt.config == {}
        assert rt.reasoning_llm is None
        assert rt.embedding_model is None
        # tool_registry is auto-created in __post_init__ (like agent_factory, tool_cache)
        assert rt.tool_registry is not None
        assert rt.skill_registry is None
        assert rt.dbase_store is None
        # hook_registry is auto-created in __post_init__ and pre-wired with the
        # agent-loop hooks (ADR-0002): mode-tag validation + workspace guard.
        assert rt.hook_registry is not None
        # Mode system defaults
        assert rt.mode == "auto"
        assert "auto" in rt.mode_config
        assert "ingest" in rt.mode_config
        assert "predict" in rt.mode_config

    def test_starts_healthy(self):
        """The runtime starts healthy (design/2_runtime_structure.md health flag)."""
        rt = DargusRuntime()
        assert rt.healthy is True
        assert rt.unhealthy_reason is None

    def test_factory_and_tool_cache_auto_wired(self):
        rt = DargusRuntime()
        assert rt.agent_factory is not None
        assert rt.agent_factory._runtime is rt
        assert isinstance(rt.tool_cache, ToolCache)

    def test_injected_factory_and_cache_kept(self):
        factory = object()
        cache = object()
        rt = DargusRuntime(agent_factory=factory, tool_cache=cache)
        assert rt.agent_factory is factory
        assert rt.tool_cache is cache

    def test_custom_config(self):
        config = {"key": "value"}
        rt = DargusRuntime(config=config)
        assert rt.config == config

    def test_set_reasoning_llm(self):
        rt = DargusRuntime()
        fake_llm = object()
        rt.reasoning_llm = fake_llm  # type: ignore[assignment]
        assert rt.reasoning_llm is fake_llm

    def test_set_embedding_model(self):
        rt = DargusRuntime()
        fake_emb = object()
        rt.embedding_model = fake_emb  # type: ignore[assignment]
        assert rt.embedding_model is fake_emb


class TestHealthFlag:
    """Health flag semantics: starts healthy, unhealthy only on failure."""

    def test_mark_unhealthy_records_reason(self):
        rt = DargusRuntime()
        rt.mark_unhealthy("D-Base inaccessible")
        assert rt.healthy is False
        assert rt.unhealthy_reason == "D-Base inaccessible"

    def test_ensure_healthy_passes_when_healthy(self):
        DargusRuntime().ensure_healthy()  # must not raise

    def test_ensure_healthy_raises_when_unhealthy(self):
        rt = DargusRuntime()
        rt.mark_unhealthy("model unavailable")
        with pytest.raises(RuntimeError, match="model unavailable"):
            rt.ensure_healthy()

    def test_shutdown_closes_tool_cache_and_marks_unhealthy(self):
        rt = DargusRuntime()
        rt.tool_cache.put("heavy", object())
        rt.shutdown()
        assert rt.healthy is False
        with pytest.raises(RuntimeError, match="closed"):
            rt.tool_cache.get("heavy")


class TestHealthCheck:
    """Tests for health_check() — presence of both model dependencies."""

    def test_healthy_when_both_models_present(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        rt.embedding_model = object()  # type: ignore[assignment]
        assert health_check(rt) is True

    def test_unhealthy_when_reasoning_llm_missing(self):
        rt = DargusRuntime()
        rt.embedding_model = object()  # type: ignore[assignment]
        assert health_check(rt) is False

    def test_unhealthy_when_embedding_model_missing(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        assert health_check(rt) is False

    def test_unhealthy_when_both_missing(self):
        assert health_check(DargusRuntime()) is False


# ------------------------------------------------------------------
# ModeSpec tests
# ------------------------------------------------------------------


class TestModeSpec:
    """Tests for ModeSpec dataclass construction."""

    def test_default_construction(self):
        ms = ModeSpec()
        assert ms.tools == []
        assert ms.skills == []
        assert ms.hooks == []
        assert ms.system_prompt == ""
        assert ms.on_enter is None
        assert ms.on_exit is None

    def test_full_construction(self):
        ms = ModeSpec(
            tools=["read_file", "dbase_query"],
            skills=["evidence_analysis"],
            hooks=["mode_tag_validation"],
            system_prompt="You are in auto mode.",
            on_enter="on_enter_hook",
            on_exit="on_exit_hook",
        )
        assert ms.tools == ["read_file", "dbase_query"]
        assert ms.skills == ["evidence_analysis"]
        assert ms.hooks == ["mode_tag_validation"]
        assert ms.system_prompt == "You are in auto mode."
        assert ms.on_enter == "on_enter_hook"
        assert ms.on_exit == "on_exit_hook"

    def test_default_mode_config_has_three_modes(self):
        config = default_mode_config()
        assert set(config.keys()) == {"auto", "ingest", "predict"}
        assert all(isinstance(v, ModeSpec) for v in config.values())

    def test_default_mode_config_has_system_prompts(self):
        config = default_mode_config()
        for mode_name in ("auto", "ingest", "predict"):
            assert config[mode_name].system_prompt, f"{mode_name} should have a system prompt"

    def test_default_mode_config_has_tools(self):
        config = default_mode_config()
        assert "switch_mode" in config["auto"].tools
        assert "read_file" in config["auto"].tools


# ------------------------------------------------------------------
# Mode switching tests
# ------------------------------------------------------------------


class TestModeSwitching:
    """Tests for DargusRuntime.switch_mode()."""

    def test_switch_to_valid_mode(self):
        rt = DargusRuntime()
        assert rt.mode == "auto"
        ok = rt.switch_mode("ingest")
        assert ok is True
        assert rt.mode == "ingest"

    def test_switch_to_unknown_mode_is_noop(self):
        rt = DargusRuntime()
        ok = rt.switch_mode("nonexistent")
        assert ok is False
        assert rt.mode == "auto"

    def test_switch_mode_fires_on_exit_hook(self):
        """on_exit hook name is in spec but no matching registered hook → no error."""
        rt = DargusRuntime()
        rt.mode_config["auto"].on_exit = "some_exit_hook"
        # Should not raise — hook not found is logged and skipped
        ok = rt.switch_mode("ingest")
        assert ok is True

    def test_switch_mode_fires_on_enter_hook(self):
        """on_enter hook name is in spec but no matching registered hook → no error."""
        rt = DargusRuntime()
        rt.mode_config["ingest"].on_enter = "some_enter_hook"
        ok = rt.switch_mode("ingest")
        assert ok is True
        assert rt.mode == "ingest"

    def test_mode_defaults_to_auto(self):
        rt = DargusRuntime()
        assert rt.mode == "auto"


# ------------------------------------------------------------------
# Mode config from YAML
# ------------------------------------------------------------------


class TestModeConfigFromConfig:
    """Tests for _mode_config_from_config()."""

    def test_empty_config_returns_defaults(self):
        result = _mode_config_from_config({})
        assert "auto" in result
        assert "ingest" in result
        assert "predict" in result

    def test_yaml_modes_override_defaults(self):
        config = {
            "modes": {
                "auto": {
                    "tools": ["custom_tool"],
                    "skills": [],
                    "hooks": [],
                }
            }
        }
        result = _mode_config_from_config(config)
        assert result["auto"].tools == ["custom_tool"]

    def test_partial_yaml_merges_with_defaults(self):
        config = {
            "modes": {
                "auto": {
                    "tools": ["override_tool"],
                }
            }
        }
        result = _mode_config_from_config(config)
        # Only tools overridden; system_prompt should come from defaults
        assert result["auto"].tools == ["override_tool"]
        assert result["auto"].system_prompt  # still has a prompt
