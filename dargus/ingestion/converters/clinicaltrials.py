"""ClinicalTrials.gov rct converter.

Turns a clinicaltrials slice ``raw.jsonl`` (provenance wrappers with a
``data.protocolSection`` payload) into ``rct`` evidence records.

Mapping:
  * conditions          -> resolve each via the disease resolver, union the
                           resulting ``mondo:`` CURIEs into ``bg.disease_id``
  * DRUG interventions  -> x-axis entity (``chembl:`` CURIE when the curated
                           drug list resolves it, else ``entity_label`` only)
  * primary outcome     -> y.type = outcome measure text, category
                           ``clinic_efficacy_primary``, value placeholder
  * study id / phase    -> ``clinical_design.study_id`` / ``phase``

Trials with no DRUG intervention, or with no resolvable condition, are
skipped with an explicit reason (``unmapped_disease:<term>`` or
``no_drug_intervention``). No sidecar fields are emitted.
"""

from __future__ import annotations

import re
from typing import Any

from dargus.ingestion.converters._entities import resolve_drug
from dargus.ingestion.converters.base import BaseConverter
from dargus.ingestion.converters.pipeline import SkipRecord
from dargus.ingestion.resolver import resolve_disease

_PHASE_MAP = {
    "PHASE1": "phase_1",
    "PHASE2": "phase_2",
    "PHASE3": "phase_3",
    "PHASE4": "phase_4",
    "EARLY_PHASE1": "phase_1",
}
_NCT_RE = re.compile(r"NCT\d{8}")


class ClinicalTrialsConverter(BaseConverter):
    """Convert ClinicalTrials raw wrappers into rct evidence records."""

    template_id = "clinicaltrials"

    def convert(self, raw: dict[str, Any]) -> list[dict[str, Any] | SkipRecord]:
        source_entry = str(raw.get("source_entry", ""))
        source_time = str(raw.get("source_time", ""))
        data = raw.get("data") or {}
        ps = data.get("protocolSection") if isinstance(data, dict) else None
        if not isinstance(ps, dict):
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="malformed_record",
                    detail="missing protocolSection",
                )
            ]

        conditions = (ps.get("conditionsModule") or {}).get("conditions") or []
        if not conditions:
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="no_condition",
                    detail="trial has no listed conditions",
                )
            ]

        disease_ids: list[str] = []
        unmapped: list[str] = []
        for cond in conditions:
            curie = resolve_disease(cond)
            if curie:
                if curie not in disease_ids:
                    disease_ids.append(curie)
            else:
                unmapped.append(str(cond))
        if not disease_ids:
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="unmapped_disease",
                    detail=";".join(unmapped[:5]),
                )
            ]

        # x-axis: first DRUG intervention (or any interventional arm)
        interventions = (ps.get("armsInterventionsModule") or {}).get("interventions") or []
        drug_int = next(
            (i for i in interventions if isinstance(i, dict) and i.get("type") == "DRUG"),
            None,
        )
        if drug_int is None:
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="no_drug_intervention",
                    detail="no DRUG-type intervention found",
                )
            ]
        drug_name = str(drug_int.get("name") or "").strip()
        drug_id, drug_label = resolve_drug(drug_name)

        # clinical_design
        design_module = ps.get("designModule") or {}
        phases = design_module.get("phases") or []
        phase = _PHASE_MAP.get(phases[0]) if phases else None
        nct = _strip_nct(source_entry)
        if not _NCT_RE.fullmatch(nct):
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="malformed_record",
                    detail=f"source_entry is not a valid NCT id: {nct!r}",
                )
            ]
        study_id = f"clinicaltrials:{nct}"
        clinical_design: dict[str, Any] = {
            "comparator_type": "no_treatment",
            "n_arms": max(1, len(interventions)),
            "population": "adults",
            "study_id": study_id,
        }
        if phase:
            clinical_design["phase"] = phase

        # y-axis: primary outcome measure (fallback to brief title)
        primary = (ps.get("outcomesModule") or {}).get("primaryOutcomes") or []
        if isinstance(primary, dict):
            primary = [v for v in primary.values() if isinstance(v, dict) and "measure" in v]
        measure = ""
        if primary:
            first = primary[0] if isinstance(primary[0], dict) else {}
            measure = str(first.get("measure") or "")[:200]
        if not measure:
            measure = str((ps.get("identificationModule") or {}).get("briefTitle") or "")[:200]

        raw_evidence = {
            "biological_level": "rct",
            "evidence_design": "descriptive",
            "xy": {"count": 1},
            "x": {
                "type": "drug",
                "value": [{"entity_id": drug_id, "entity_label": drug_label or drug_name}],
            },
            "y": {
                "type": measure,
                "category": "clinic_efficacy_primary",
                "value": [1.0],
                "to_basis": "absolute",
            },
            "bg": {"disease_id": disease_ids, "drugs": [], "genes": []},
            "clinical_design": clinical_design,
            "source_entry": source_entry,
            "source_time": source_time,
        }
        return [raw_evidence]


def _strip_nct(source_entry: str) -> str:
    """Return the bare NCT id from a ``clinicaltrials:`` source_entry."""
    return source_entry.split(":", 1)[-1] if ":" in source_entry else source_entry
