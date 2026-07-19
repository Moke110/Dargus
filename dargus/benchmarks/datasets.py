"""Benchmark dataset loaders."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from dargus.benchmarks.config import get_dataset_config


class BenchmarkDataset:
    """Load a benchmark dataset and normalize it for Dargus training/evaluation."""

    def __init__(self, name: str, data_root: str = "data", config: dict[str, Any] | None = None):
        self.name = name
        self.data_root = Path(data_root)
        self.config = config or get_dataset_config(name)

    def load(self, split: str = "train") -> pd.DataFrame:
        """Load a split of the benchmark dataset.

        Returns a DataFrame with canonical columns:
        ``drug_id``, ``disease_id``, ``endpoint``, ``label``.
        """
        split_path = self.config[split]
        full_path = self.data_root / self.config["type"] / self.config["category"] / split_path
        df = pd.read_csv(full_path)
        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        endpoint = self.config["endpoint"]
        label_column = self.config.get("label_column", "label")

        normalized = pd.DataFrame(
            {
                "drug_id": df["drugs"].apply(self._first_list_item),
                "disease_id": df["diseases"].apply(self._first_list_item),
                "endpoint": endpoint,
                "label": df[label_column].astype(int),
            }
        )
        # Preserve optional columns that downstream loaders may use.
        for col in ["nctid", "smiless", "phase", "status"]:
            if col in df.columns:
                normalized[col] = df[col]
        return normalized

    @staticmethod
    def _first_list_item(value: Any) -> str:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return str(parsed[0])
                return str(parsed)
            except (ValueError, SyntaxError):
                return value
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        return str(value)
