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
            entity_id = drug if ":" in drug else f"chembl:{drug}"
            entity: dict[str, Any] = {"entity_id": entity_id, "entity_label": drug}
            smiles = row.get("SMILES", row.get("smiles", None))
            if pd.notna(smiles):
                entity["smiles"] = str(smiles)
            rows.append(
                {
                    "x": {"type": "drug", "value": [entity]},
                    "y": {
                        "type": self.assay_name,
                        "category": "pk_adme",
                        "value": [float(row["Y"])],
                    },
                }
            )
        return rows
