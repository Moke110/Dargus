"""OpenFDA label epi converter.

Turns an OpenFDA slice ``raw.jsonl`` (provenance wrappers with
``data.drug`` + ``data.indications`` prose) into ``epi`` evidence records.

Mapping:
  * the approved-indications prose -> resolve via the disease resolver
    (substring match against the curated registry); the resulting ``mondo:``
    CURIE becomes ``bg.disease_id``
  * the label drug -> x-axis entity; a ``chembl:`` CURIE when the curated
    drug list resolves it, else ``entity_label`` only (conservatism rule:
    never invent a CURIE)
  * y-axis records the indication statement as the measured outcome

Records with a 404 status, or whose indications prose cannot be resolved to
a disease, are skipped with an explicit reason. No sidecar fields.
"""

from __future__ import annotations

from typing import Any

from dargus.ingestion.converters._entities import resolve_drug
from dargus.ingestion.converters.base import BaseConverter
from dargus.ingestion.converters.pipeline import SkipRecord
from dargus.ingestion.resolver import resolve_disease


class OpenFDAConverter(BaseConverter):
    """Convert OpenFDA label raw wrappers into epi evidence records."""

    template_id = "openfda"

    def convert(self, raw: dict[str, Any]) -> list[dict[str, Any] | SkipRecord]:
        source_entry = str(raw.get("source_entry", ""))
        source_time = str(raw.get("source_time", ""))
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            return [self._skip(source_entry, "malformed_record", "data is not an object")]

        if "status" in data:
            return [
                self._skip(
                    source_entry,
                    "no_fda_label",
                    detail=f"drug not found in FDA OpenFDA: {str(data.get('status'))[:80]}",
                )
            ]

        drug = str(data.get("drug") or "").strip()
        indications = str(data.get("indications") or "").strip()
        if not drug:
            return [self._skip(source_entry, "malformed_record", "missing drug name")]
        if not indications:
            return [self._skip(source_entry, "no_indications", "label has no indications text")]

        disease_id = resolve_disease(indications)
        if not disease_id:
            return [
                self._skip(
                    source_entry,
                    "unmapped_disease",
                    detail=f"could not resolve indication prose for {drug}",
                )
            ]

        drug_id, drug_label = resolve_drug(drug)
        raw_evidence = {
            "biological_level": "epi",
            "evidence_design": "observational_association",
            "xy": {"count": 1},
            "x": {
                "type": "drug",
                "value": [{"entity_id": drug_id, "entity_label": drug_label or drug}],
            },
            "y": {
                "type": "fda_approved_indication",
                "category": "clinic_efficacy_primary",
                "value": [1.0],
                "to_basis": "absolute",
                "direction": "beneficial",
            },
            "bg": {"disease_id": [disease_id], "drugs": [], "genes": []},
            "clinical_design": {
                "comparator_type": "no_treatment",
                "population": "adults",
            },
            "source_entry": source_entry,
            "source_time": source_time,
        }
        return [raw_evidence]

    @staticmethod
    def _skip(source_entry: str, reason: str, detail: str) -> SkipRecord:
        return SkipRecord(
            source_entry=source_entry,
            source="openfda",
            reason=reason,
            detail=detail,
        )
