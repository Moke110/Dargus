from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dargus.ingestion.converters.base import BaseConverter


class GdscConverter(BaseConverter):
    template_id = "cell_viability_assay_v1"

    def convert(self, path: Path) -> list[dict[str, Any]]:
        df = pd.read_csv(path)
        required = {"DRUG_ID", "CELL_LINE_NAME", "TCGA_DESC", "LN_IC50"}
        if not required.issubset(df.columns):
            return []

        rows = []
        for _, row in df.iterrows():
            try:
                readout = float(row["LN_IC50"])
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "biological_level": "cellular",
                    "drug_id": str(row["DRUG_ID"]),
                    "cell_line_id": str(row["CELL_LINE_NAME"]),
                    "disease_id": str(row["TCGA_DESC"]).upper(),
                    "assay_type": "gdsc2_ln_ic50",
                    "readout": readout,
                }
            )
        return rows
