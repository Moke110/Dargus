from __future__ import annotations

import numpy as np

from dargus.dbase import DBase

LEVEL_ORDER = ["molecular", "cellular", "exvivo", "animal", "clinical"]


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

        records = self.dbase.query(disease_id=disease_id)
        group_counter = 0
        for rec in records:
            if rec.template_id not in self.dbase._templates:
                continue
            schema = self.dbase._templates[rec.template_id]

            # Check drug match
            try:
                drug_idx = schema.field_index("drug_id")
            except KeyError:
                continue
            indices = rec.sparse_vector.get("indices", [])
            values = rec.sparse_vector.get("values", [])
            if drug_idx not in indices:
                continue
            drug_val = values[indices.index(drug_idx)]
            drug_term = self.dbase.vocab.reverse_lookup(
                schema.field_def("drug_id").vocabulary_ref or "drug_id", int(drug_val)
            )
            if drug_term not in drug_ids:
                continue

            # Endpoint filter
            if endpoints:
                try:
                    endpoint_idx = schema.field_index("endpoint")
                    if endpoint_idx in indices:
                        endpoint_val = values[indices.index(endpoint_idx)]
                        endpoint_term = self.dbase.vocab.reverse_lookup(
                            schema.field_def("endpoint").vocabulary_ref or "endpoint",
                            int(endpoint_val),
                        )
                        if endpoint_term not in endpoints:
                            continue
                except KeyError:
                    pass

            # Read fold_change if present, else readout
            value = None
            for field_name in ["fold_change", "readout"]:
                try:
                    idx = schema.field_index(field_name)
                    if idx in indices:
                        value = float(values[indices.index(idx)])
                        break
                except KeyError:
                    continue
            if value is None:
                continue

            level = self._level_for_record(rec)
            y_list.append(value)
            level_list.append(LEVEL_ORDER.index(level))
            group_list.append(group_counter)
            record_ids.append(rec.record_id)
            group_counter += 1

        return (
            np.array(y_list, dtype=float),
            np.array(level_list, dtype=int),
            np.array(group_list, dtype=int),
            record_ids,
        )

    def _level_for_record(self, rec) -> str:
        schema = self.dbase._templates.get(rec.template_id)
        if schema is None:
            return "molecular"
        try:
            idx = schema.field_index("biological_level")
        except KeyError:
            return "molecular"
        indices = rec.sparse_vector.get("indices", [])
        values = rec.sparse_vector.get("values", [])
        if idx not in indices:
            return "molecular"
        val = int(values[indices.index(idx)])
        field = schema.field_def("biological_level")
        vocab = field.vocabulary
        if val < len(vocab):
            return vocab[val]
        return "molecular"
