"""Dataset splitters for benchmark evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class StratifiedDrugDiseaseEndpointSplitter:
    """Split a DataFrame so no (drug, disease, endpoint) group appears in both splits."""

    def __init__(self, test_size: float = 0.2, random_state: int | None = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"drug_id", "disease_id", "endpoint"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        groups = df.groupby(["drug_id", "disease_id", "endpoint"], sort=False).ngroup()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
