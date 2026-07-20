from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

from dargus.experts.types import ExtractedInstance


class ConverterRegistry:
    """Registry of known file patterns and their column mappings."""

    def __init__(self):
        self._entries: list[dict[str, Any]] = []

    def register(self, **entry: Any) -> None:
        self._entries.append(entry)

    def match(self, path: Path) -> dict[str, Any] | None:
        try:
            df = self._read_head(path)
        except Exception:  # noqa: BLE001
            return None
        columns = set(df.columns)
        for entry in self._entries:
            pattern = entry.get("match", {}).get("path_pattern", "")
            if not fnmatch.fnmatch(path.name, pattern):
                continue
            required = set(entry.get("match", {}).get("columns_required", []))
            if required.issubset(columns):
                return entry
        return None

    def convert_file(
        self, path: Path, entry: dict[str, Any] | None = None
    ) -> list[ExtractedInstance]:
        if entry is None:
            entry = self.match(path)
        if entry is None:
            return []

        df = self._read(path)
        mapping = entry["field_mapping"]
        template_id = entry["template_id"]
        df_columns = set(df.columns)
        instances: list[ExtractedInstance] = []
        for row_idx, row in df.iterrows():
            raw_fields: dict[str, Any] = {}
            for target_field, source_col in mapping.items():
                if target_field == "biological_level":
                    continue
                if source_col in df_columns:
                    if pd.notna(row[source_col]):
                        raw_fields[target_field] = row[source_col]
                else:
                    raw_fields[target_field] = source_col
            instances.append(
                ExtractedInstance(
                    template_id=template_id,
                    raw_fields=raw_fields,
                    source_file=str(path),
                    source_row=int(row_idx),
                    extraction_confidence="high",
                )
            )
        return instances

    def _read_head(self, path: Path) -> pd.DataFrame:
        return self._read(path, nrows=3)

    def _read(self, path: Path, nrows: int | None = None) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows, on_bad_lines="skip")
        if suffix in {".tsv", ".tab"}:
            return pd.read_csv(path, sep="\t", nrows=nrows, on_bad_lines="skip")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=nrows)
        raise ValueError(f"Unsupported suffix: {suffix}")
