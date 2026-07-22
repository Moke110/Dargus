"""ToolRegistry — load registry.yaml and provide domain/level-based tool lookup."""

from __future__ import annotations

from pathlib import Path

import yaml

from dargus.tools.base import Tool, ToolParam


class ToolRegistry:
    """Load registry.yaml and provide name/domain/level-based tool lookup."""

    def __init__(self, yaml_path: Path | str | None = None):
        if yaml_path is None:
            yaml_path = Path(__file__).resolve().parent / "registry.yaml"
        self._path = Path(yaml_path)
        self._tools: dict[str, Tool] = {}
        self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for name, entry in (data.get("tools") or {}).items():
            params = []
            for pname, pdef in (entry.get("parameters") or {}).items():
                params.append(
                    ToolParam(
                        name=pname,
                        type=pdef.get("type", "string"),
                        required=pdef.get("required", False),
                        default=pdef.get("default"),
                        description=pdef.get("description", ""),
                        enum=pdef.get("enum"),
                    )
                )
            tool = Tool(
                name=name,
                description=entry.get("description", ""),
                parameters=params,
                output=entry.get("output", {}),
                timeout_ms=entry.get("timeout_ms", 10_000),
                fallback=entry.get("fallback", "empty_list"),
            )
            tool._domain = entry.get("domain", "")
            tool._biological_levels = entry.get("biological_levels", [])
            self._tools[name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def list_by_domain(self, domain: str) -> list[Tool]:
        return [t for t in self._tools.values() if getattr(t, "_domain", "") == domain]

    def list_by_level(self, biological_level: str) -> list[Tool]:
        return [
            t
            for t in self._tools.values()
            if biological_level in getattr(t, "_biological_levels", [])
        ]
