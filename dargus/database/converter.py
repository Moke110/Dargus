"""Converters from common formats to DataMaster sample records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def csv_to_samples(path: str | Path, **defaults: Any) -> list[dict[str, Any]]:
    """Read a CSV file and return sample dicts."""
    df = pd.read_csv(path)
    return _dataframe_to_samples(df, **defaults)


def excel_to_samples(path: str | Path, **defaults: Any) -> list[dict[str, Any]]:
    """Read an Excel file and return sample dicts."""
    df = pd.read_excel(path)
    return _dataframe_to_samples(df, **defaults)


def _dataframe_to_samples(df: pd.DataFrame, **defaults: Any) -> list[dict[str, Any]]:
    samples = []
    for _, row in df.iterrows():
        record = {**defaults, **row.dropna().to_dict()}
        samples.append(record)
    return samples
