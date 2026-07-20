from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class ExvivoToolRAG(ToolRAG):
    """ToolRAG for ex-vivo data: tissue slices, primary cell cultures."""

    def __init__(self):
        super().__init__("exvivo")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass
