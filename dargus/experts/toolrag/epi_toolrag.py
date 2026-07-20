from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class EpiToolRAG(ToolRAG):
    """ToolRAG for epidemiological data: population studies, real-world evidence."""

    def __init__(self):
        super().__init__("epi")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass
