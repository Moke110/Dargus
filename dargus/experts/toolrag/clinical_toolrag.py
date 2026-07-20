from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class ClinicalToolRAG(ToolRAG):
    """ToolRAG for clinical-level data: trial results, patient cohorts."""

    def __init__(self):
        super().__init__("clinical")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass
