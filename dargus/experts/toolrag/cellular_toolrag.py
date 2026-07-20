from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class CellularToolRAG(ToolRAG):
    """ToolRAG for cellular-level data: cell-based assays, proliferation, toxicity."""

    def __init__(self):
        super().__init__("cellular")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass
