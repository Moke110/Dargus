"""ToolRegistry — programmatic Tool registration and lookup.

The task-specific ``registry.yaml`` definitions were removed with the
task-specific code. Tools are now registered programmatically (e.g. the
runtime's file Tools), so the registry is a plain name-keyed store with no
domain/level metadata.
"""

from __future__ import annotations

from dargus.tools.base import Tool


class ToolRegistry:
    """Name-keyed registry of Tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a Tool instance.

        Validates that any Tool declaring ``side_effect="write"`` holds a
        WorkspaceGuard and declares at least one ``"path"``-typed parameter.
        The failure is loud at registration time, not silent at call time.
        """
        if tool.side_effect == "write":
            if tool._guard is None:
                raise ValueError(
                    f"Tool '{tool.name}' declares side_effect='write' "
                    f"but has no WorkspaceGuard injected"
                )
            path_params = [p for p in tool.parameters if p.type == "path"]
            if not path_params:
                raise ValueError(
                    f"Tool '{tool.name}' declares side_effect='write' "
                    f"but has no 'path'-typed parameters"
                )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())
