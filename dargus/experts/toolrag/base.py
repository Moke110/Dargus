from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dargus.experts.types import ExtractionReport


class ToolRAG(ABC):
    """Base class for level-specific tool/knowledge retrieval."""

    def __init__(self, level_name: str):
        self.level_name = level_name

    @abstractmethod
    def extract(self, raw_data_dir: str) -> ExtractionReport:
        """Scan raw_data_dir and extract instances for this level."""
        ...

    def _discover_files(self, raw_data_dir: str) -> list[Path]:
        path = Path(raw_data_dir)
        if not path.exists():
            return []
        return [p for p in path.rglob("*") if p.is_file()]
