"""D-Base record adapter v0.15.0 — evidence dict API."""

from __future__ import annotations

import numpy as np

from dargus.dbase import DBase

LEVEL_ORDER = ["molecular", "cellular", "exvivo", "animal", "rct", "epi"]


class RecordsAdapter:
    def __init__(self, dbase: DBase):
        self.dbase = dbase

    def to_arrays(
        self,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        y_list: list[float] = []
        level_list: list[int] = []
        group_list: list[int] = []
        record_ids: list[str] = []

        records = self.dbase.read_shards()
        group_counter = 0
        for rec in records:
            # Drug match via interventions
            interventions = rec.get("interventions", [])
            primary = next((i for i in interventions if i.get("role") == "primary"), None)
            rec_drug = (primary or {}).get("entity_id", "")
            if rec_drug not in drug_ids:
                continue

            # Disease match
            if rec.get("disease_id") != disease_id:
                continue

            # Endpoint filter
            if endpoints:
                rec_ep = rec.get("readout_type", "") or rec.get("endpoint", "")
                if rec_ep not in endpoints:
                    continue

            # Readout value
            value = rec.get("readout_value") or rec.get("fold_change")
            if value is None:
                continue

            # Level
            level = rec.get("biological_level", "molecular")
            level_idx = LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0

            y_list.append(float(value))
            level_list.append(level_idx)
            group_list.append(group_counter)
            record_ids.append(rec.get("evidence_id", ""))
            group_counter += 1

        return (
            np.array(y_list, dtype=float),
            np.array(level_list, dtype=int),
            np.array(group_list, dtype=int),
            record_ids,
        )
