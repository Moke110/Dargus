from __future__ import annotations

from abc import ABC
from pathlib import Path

from dargus.experts.types import ExtractedInstance, ExtractionReport


class ToolRAG(ABC):
    """Base class for level-specific tool/knowledge retrieval."""

    def __init__(self, level_name: str):
        self.level_name = level_name

    def extract(self, raw_data_dir: str) -> ExtractionReport:
        """Scan raw_data_dir and extract instances for this level."""
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

    def _discover_files(self, raw_data_dir: str) -> list[Path]:
        path = Path(raw_data_dir)
        if not path.exists():
            return []
        return [p for p in path.rglob("*") if p.is_file()]
