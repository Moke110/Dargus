from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dargus.ingestion.converters.base import BaseConverter


class TdcDtiConverter(BaseConverter):
    template_id = "dti_assay_v1"

    def __init__(self, assay_name: str = "affinity"):
        self.assay_name = assay_name

    def convert(self, path: Path) -> list[dict[str, Any]]:
        df = pd.read_csv(path, sep="\t")
        rows = []
        for _, row in df.iterrows():
            drug = str(row.get("Drug", ""))
            target = str(row.get("Target", ""))
            if not drug or not target:
                continue
            rows.append(
                {
                    "drug_id": drug,
                    "target_id": target,
                    "assay_type": self.assay_name,
                    "readout": float(row["Y"]),
                }
            )
        return rows
