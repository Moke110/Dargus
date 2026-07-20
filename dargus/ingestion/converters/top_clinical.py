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
            phase = self._normalize_phase(str(row.get("phase", "")))
            for disease in diseases:
                for drug in drugs:
                    rows.append(
                        {
                            "biological_level": "clinical",
                            "drug_id": drug,
                            "disease_id": disease,
                            "endpoint": "trial_success",
                            "fold_change": float(row["label"]),
                            "phase": phase,
                        }
                    )
        return rows

    def _parse_list(self, value: Any) -> list[str]:
        if pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except (ValueError, SyntaxError):
            pass
        return []

    def _normalize_phase(self, phase: str) -> str:
        lower = phase.lower()
        if "phase 3" in lower or "phase iii" in lower:
            return "III"
        if "phase 2" in lower or "phase ii" in lower:
            return "II"
        if "phase 1" in lower or "phase i" in lower:
            return "I"
        return ""
