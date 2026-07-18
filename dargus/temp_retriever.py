from __future__ import annotations

import uuid
from typing import Any

from dargus.dbase import DBase, TemplateRecord

_UNIT_SUFFIXES = frozenset({"nm", "um", "mm", "mgml", "μm"})


class TempRetriever:
    """Maps raw inputs to D-Base TemplateRecords. Sole writer to D-Base."""

    def __init__(self, dbase: DBase):
        self.dbase = dbase

    def fill_template(
        self,
        raw_input: dict[str, Any],
        source_metadata: dict[str, Any],
        suggested_template: str | None = None,
    ) -> TemplateRecord:
        template_id = suggested_template or self._match_template(raw_input)
        schema = self.dbase.get_template(template_id)

        indices: list[int] = []
        values: list[float] = []

        for field in schema.fields:
            val = self._extract_field(field, raw_input)
            if val is None:
                continue
            idx = schema.field_index(field.name)
            indices.append(idx)
            values.append(int(val) if field.type == "factor" else float(val))

        return TemplateRecord(
            template_id=template_id,
            record_id=f"rec_{uuid.uuid4().hex[:12]}",
            source=source_metadata,
            sparse_vector={"indices": indices, "values": values},
            provenance_note=raw_input.get("note", ""),
        )

    def _match_template(self, raw_input: dict[str, Any]) -> str:
        # MVP: simple keyword matching. Later: LLM-assisted.
        def _normalize_key(key: str) -> str:
            key = key.lower()
            if "_" in key:
                base, _, suffix = key.rpartition("_")
                if suffix in _UNIT_SUFFIXES:
                    return base
            return key

        normalized_keys = {_normalize_key(k) for k in raw_input}

        if {"ic50", "ki", "kd"} & normalized_keys:
            candidate = "in_vitro_kinase_inhibition_v1"
            try:
                self.dbase.get_template(candidate)
                return candidate
            except KeyError:
                pass
        if {"cell_viability", "cc50", "ec50"} & normalized_keys:
            candidate = "cell_viability_assay_v1"
            try:
                self.dbase.get_template(candidate)
                return candidate
            except KeyError:
                pass

        available = sorted(p.stem for p in self.dbase.templates_dir.glob("*.yaml"))
        if available:
            return available[0]
        raise RuntimeError("No templates registered in D-Base")

    def _extract_field(self, field, raw_input: dict[str, Any]) -> int | float | None:
        name = field.name
        synonyms = {
            "drug_id": ["drug", "compound", "molecule"],
            "target_id": ["target", "gene", "protein"],
            "disease_id": ["disease", "indication"],
            "assay_type": ["assay", "method"],
            "readout": [
                "ic50",
                "ic50_nM",
                "ic50_nm",
                "ki",
                "ki_nM",
                "ki_nm",
                "kd",
                "kd_nM",
                "kd_nm",
                "ec50",
                "ec50_nM",
                "ec50_nm",
                "cc50",
                "cc50_nM",
                "cc50_nm",
                "value",
            ],
            "log_pvalue": ["p_value", "pvalue", "p"],
        }
        keys_to_try = [name] + synonyms.get(name, [])
        for key in keys_to_try:
            if key in raw_input:
                raw_val = raw_input[key]
                if field.type == "factor":
                    str_val = str(raw_val)
                    if field.vocabulary:
                        # Inline vocabulary: factor value is list index
                        if str_val in field.vocabulary:
                            return field.vocabulary.index(str_val)
                        raise ValueError(
                            f"Value {str_val!r} not in inline vocabulary "
                            f"for {field.name}: {field.vocabulary}"
                        )
                    vocab_ref = field.vocabulary_ref or name
                    return self.dbase.vocab.get_or_create(vocab_ref, str_val)
                return raw_val
        return None

    def create_field_request(
        self, field_name: str, template_id: str, reason: str
    ) -> dict[str, Any]:
        return {
            "type": "field_extension_request",
            "template_id": template_id,
            "field_name": field_name,
            "reason": reason,
            "options": [
                "add_new_field",
                "map_to_existing_field",
                "skip_record",
            ],
        }

    def write_record(self, record: TemplateRecord) -> None:
        self.dbase.add_record(record)
