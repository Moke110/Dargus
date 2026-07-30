"""Tests for Tool schema extensions — path type, side_effect, registry validation (#55)."""

from __future__ import annotations

import pytest

from dargus.tools.base import Tool, ToolParam
from dargus.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# ToolParam "path" type
# ---------------------------------------------------------------------------


class TestPathParamType:
    def test_path_type_accepted_on_param(self):
        p = ToolParam(name="file", type="path", required=True, description="input file")
        assert p.type == "path"
        assert p.name == "file"
        assert p.required is True

    def test_path_param_in_tool(self):
        t = Tool(
            name="test_tool",
            description="test",
            parameters=[
                ToolParam(name="input_file", type="path", required=True, description="file path"),
            ],
            output={"type": "object"},
        )
        assert any(p.type == "path" for p in t.parameters)


# ---------------------------------------------------------------------------
# Tool side_effect declaration
# ---------------------------------------------------------------------------


class TestToolSideEffect:
    def test_default_side_effect_is_none(self):
        t = Tool(
            name="noop",
            description="test",
            parameters=[],
            output={"type": "object"},
        )
        assert t.side_effect == "none"

    def test_read_side_effect(self):
        t = Tool(
            name="reader",
            description="reads files",
            parameters=[ToolParam(name="f", type="path", description="file")],
            output={"type": "string"},
            side_effect="read",
        )
        assert t.side_effect == "read"

    def test_write_side_effect(self):
        t = Tool(
            name="writer",
            description="writes files",
            parameters=[ToolParam(name="dest", type="path", description="output file")],
            output={"type": "object"},
            side_effect="write",
        )
        assert t.side_effect == "write"


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------


class _FakeGuard:
    """Duck-typed WorkspaceGuard for registry validation tests."""


class TestRegistryValidation:
    def test_register_write_tool_without_guard_raises(self):
        reg = ToolRegistry()
        tool = Tool(
            name="bad_writer",
            description="no guard",
            parameters=[ToolParam(name="f", type="path", description="file")],
            output={"type": "object"},
            side_effect="write",
            # _guard is None by default
        )
        with pytest.raises(ValueError, match="no WorkspaceGuard"):
            reg.register(tool)

    def test_register_write_tool_without_path_param_raises(self):
        reg = ToolRegistry()
        tool = Tool(
            name="bad_writer",
            description="no path param",
            parameters=[ToolParam(name="x", type="string", description="text")],
            output={"type": "object"},
            side_effect="write",
        )
        tool.inject_guard(_FakeGuard())
        with pytest.raises(ValueError, match="no 'path'-typed parameters"):
            reg.register(tool)

    def test_register_write_tool_with_guard_and_path_param_succeeds(self):
        reg = ToolRegistry()
        tool = Tool(
            name="good_writer",
            description="has guard + path param",
            parameters=[ToolParam(name="dest", type="path", description="file")],
            output={"type": "object"},
            side_effect="write",
        )
        tool.inject_guard(_FakeGuard())
        reg.register(tool)
        assert reg.get("good_writer") is tool

    def test_register_none_read_tools_unaffected(self):
        reg = ToolRegistry()
        t1 = Tool(
            name="reader",
            description="read tool",
            parameters=[ToolParam(name="f", type="path")],
            output={"type": "string"},
            side_effect="read",
        )
        t2 = Tool(
            name="noop",
            description="no side effect",
            parameters=[ToolParam(name="x", type="string")],
            output={"type": "object"},
        )
        reg.register(t1)
        reg.register(t2)
        assert reg.get("reader") is t1
        assert reg.get("noop") is t2

    def test_register_replaces_existing_tool(self):
        """Subsequent registration of the same name replaces the prior."""
        reg = ToolRegistry()
        t1 = Tool(name="t", description="first", parameters=[], output={})
        t2 = Tool(name="t", description="second", parameters=[], output={})
        reg.register(t1)
        reg.register(t2)
        assert reg.get("t") is t2


# ---------------------------------------------------------------------------
# Regression: existing registry.yaml entries still load
# ---------------------------------------------------------------------------


class TestRegistryYamlLoad:
    def test_all_default_tools_load(self):
        """Every existing tool in the default registry.yaml must load without error."""
        reg = ToolRegistry()
        tools = reg.list_all()
        assert len(tools) > 0
        for t in tools:
            assert isinstance(t.side_effect, str)
            assert t.side_effect in ("none", "read", "write")
            # All params have valid types
            for p in t.parameters:
                assert p.type in (
                    "string",
                    "integer",
                    "float",
                    "boolean",
                    "array",
                    "object",
                    "path",
                )
