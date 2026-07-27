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
            drug_id = str(row["DRUG_ID"])
            entity_id = drug_id if ":" in drug_id else f"chembl:{drug_id}"
            rows.append(
                {
                    "biological_level": "cellular",
                    "x": {
                        "type": "drug",
                        "value": [{"entity_id": entity_id, "entity_label": drug_id}],
                    },
                    "y": {
                        "type": "ln_ic50",
                        "category": "pk_adme",
                        "value": [readout],
                        "assay": "gdsc2_ln_ic50",
                    },
                    "bg": {"disease_id": [str(row["TCGA_DESC"]).upper()]},
                    "cell_line_id": str(row["CELL_LINE_NAME"]),
                }
            )
        return rows
