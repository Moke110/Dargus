from __future__ import annotations

from pathlib import Path

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry
from dargus.experts.types import ExtractionReport, ExtractedInstance


class CellularToolRAG(ToolRAG):
    """ToolRAG for cellular-level data: cell-based assays, proliferation, toxicity."""

    def __init__(self):
        super().__init__("cellular")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        """Register known converters (empty until Task 4)."""
        pass

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        files = self._discover_files(raw_data_dir)
        selected: list[str] = []
        instances: list[ExtractedInstance] = []
        source_types: dict[str, int] = {}
        notes: list[str] = []

        for path in files:
            entry = self.registry.match(path)
            if entry is None:
                continue
            selected.append(str(path))
            name = entry.get("name", path.name)
            file_instances = self.registry.convert_file(path, entry)
            source_types[name] = source_types.get(name, 0) + len(file_instances)
            instances.extend(file_instances)

        return ExtractionReport(
            level=self.level_name,
            files_considered=[str(p) for p in files],
            files_selected=selected,
            source_types=source_types,
            instances=instances,
            notes=notes,
        )
