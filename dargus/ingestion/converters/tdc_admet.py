from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dargus.ingestion.converters.base import BaseConverter


class TdcAdmetConverter(BaseConverter):
    template_id = "admet_assay_v1"

    def __init__(self, assay_name: str):
        self.assay_name = assay_name

    def convert(self, path: Path) -> list[dict[str, Any]]:
        try:
            df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
        except pd.errors.ParserError:
            return []
        rows = []
        for _, row in df.iterrows():
            drug = str(row.get("Drug", row.get("Drug_ID", "")))
            if not drug:
                continue
            result: dict[str, Any] = {
                "drug_id": drug,
                "assay_type": self.assay_name,
                "readout": float(row["Y"]),
            }
            smiles = row.get("SMILES", row.get("smiles", None))
            if pd.notna(smiles):
                result["smiles"] = str(smiles)
            rows.append(result)
        return rows
