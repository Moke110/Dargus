"""Tool base — standardized executable wrapper with typed I/O schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParam:
    name: str
    type: str  # "string" | "integer" | "float" | "boolean" | "array" | "object" | "path"
    required: bool = False
    default: Any = None
    description: str = ""
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParam]
    output: dict[str, Any]  # JSON Schema-style output description
    timeout_ms: int = 10_000
    fallback: str = "empty_list"  # "empty_list" | "null_result" | "skip" | "error"
    side_effect: str = "none"  # "none" | "read" | "write"
    _impl: Callable | None = field(default=None, repr=False)
    _guard: Any | None = field(default=None, repr=False)  # WorkspaceGuard (forward ref)

    def execute(self, **kwargs: Any) -> Any:
        if self._impl is None:
            raise NotImplementedError(f"Tool '{self.name}' has no implementation bound")
        return self._impl(**kwargs)

    def bind(self, impl: Callable) -> None:
        self._impl = impl

    def inject_guard(self, guard: Any) -> None:
        """Receive the runtime's WorkspaceGuard (DI seam)."""
        self._guard = guard

    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def required_params(self) -> list[str]:
        return [p.name for p in self.parameters if p.required]
