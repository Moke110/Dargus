from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from dargus.ingestion.converters.base import BaseConverter


class TopClinicalConverter(BaseConverter):
    template_id = "clinical_trial_outcome_v1"

    def convert(self, path: Path) -> list[dict[str, Any]]:
        df = pd.read_csv(path)
        required = {"diseases", "drugs", "label", "phase"}
        if not required.issubset(df.columns):
            return []

        rows = []
        for _, row in df.iterrows():
            diseases = self._parse_list(row.get("diseases"))
            drugs = self._parse_list(row.get("drugs"))
            if not diseases or not drugs:
                continue
            try:
                label_val = float(row["label"])
            except (ValueError, TypeError):
                continue
            phase = self._normalize_phase(str(row.get("phase", "")))
            for disease in diseases:
                for drug in drugs:
                    entity_id = drug if ":" in drug else f"chembl:{drug}"
                    rows.append(
                        {
                            "biological_level": "rct",
                            "x": {
                                "type": "drug",
                                "value": [{"entity_id": entity_id, "entity_label": drug}],
                            },
                            "y": {
                                "type": "trial_success",
                                "category": "clinic_efficacy_primary",
                                "value": [label_val],
                            },
                            "bg": {"disease_id": [disease]},
                            "clinical_design": {"phase": phase} if phase else {},
                        }
                    )
        return rows

    def _parse_list(self, value: Any) -> list[str]:
        if pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return [str(x).strip().strip('"').strip("'") for x in parsed if x]
        except (ValueError, SyntaxError):
            pass
        # Fallback: comma-separated bare string
        parts = [p.strip().strip('"').strip("'") for p in str(value).split(",") if p.strip()]
        return parts

    def _normalize_phase(self, phase: str) -> str:
        lower = phase.lower().strip()
        mapping = {
            "1": "I",
            "i": "I",
            "2": "II",
            "ii": "II",
            "3": "III",
            "iii": "III",
            "4": "IV",
            "iv": "IV",
        }
        for part in lower.split():
            if part.startswith("phase"):
                continue
            if part in mapping:
                return mapping[part]
        return ""
