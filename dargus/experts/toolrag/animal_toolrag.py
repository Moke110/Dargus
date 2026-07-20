from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class AnimalToolRAG(ToolRAG):
    """ToolRAG for animal-level data: in vivo PK/PD, efficacy studies."""

    def __init__(self):
        super().__init__("animal")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass
