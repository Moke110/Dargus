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
        for opts in [
            {"sep": "\t", "on_bad_lines": "skip"},
            {"sep": "\t", "on_bad_lines": "skip", "quoting": 3},
            {"on_bad_lines": "skip"},
        ]:
            try:
                df = pd.read_csv(path, **opts)
                break
            except pd.errors.ParserError:
                continue
        else:
            return []

        drug_col = self._find_column(df, ["Drug", "drug", "ID1", "compound", "ligand", "SMILES"])
        target_col = self._find_column(df, ["Target", "target", "ID2", "protein", "gene"])
        readout_col = self._find_column(
            df, ["Y", "pKd", "pKi", "pIC50", "score", "Kd", "Ki", "IC50"]
        )

        rows = []
        for _, row in df.iterrows():
            drug = str(row.get(drug_col, "")) if drug_col else ""
            target = str(row.get(target_col, "")) if target_col else ""
            if not drug or not target:
                continue
            try:
                readout = float(row[readout_col]) if readout_col else 0.0
            except (ValueError, TypeError):
                continue
            entity_id = drug if ":" in drug else f"chembl:{drug}"
            rows.append(
                {
                    "x": {
                        "type": "drug",
                        "value": [{"entity_id": entity_id, "entity_label": drug}],
                    },
                    "y": {
                        "type": self.assay_name,
                        "category": "pk_adme",
                        "value": [readout],
                    },
                    "bg": {"genes": [target]},
                }
            )
        return rows

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
