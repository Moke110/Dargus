"""ClinVar variant-disease association converter.

Turns a ClinVar slice ``raw.jsonl`` (provenance wrappers with a
``data.formatted_results.result`` esummary payload) into ``epi`` evidence
records describing variant-disease associations.

Mapping:
  * disease -> prefer the variant's ``germline_classification.trait_set[*]
    .trait_xrefs`` MONDO id when present (highest confidence); else resolve
    the trait name via the disease resolver. Union the resulting ``mondo:``
    CURIEs into ``bg.disease_id``.
  * gene    -> ``bg.genes`` + x-axis ``gene`` entity (``entity_label`` only —
    gene symbols are not registered CURIE prefixes here, so no CURIE is
    invented)
  * variant -> y-axis ``type`` carries the variant title (HGVS-style)

Variants with no resolvable disease (e.g. trait "not specified") are skipped
with an explicit ``unmapped_disease`` reason. No sidecar fields.
"""

from __future__ import annotations

from typing import Any

from dargus.ingestion.converters.base import BaseConverter
from dargus.ingestion.converters.pipeline import SkipRecord
from dargus.ingestion.resolver import resolve_disease


class ClinVarConverter(BaseConverter):
    """Convert ClinVar raw wrappers into epi evidence records."""

    template_id = "clinvar"

    def convert(self, raw: dict[str, Any]) -> list[dict[str, Any] | SkipRecord]:
        source_entry = str(raw.get("source_entry", ""))
        source_time = str(raw.get("source_time", ""))
        data = raw.get("data") or {}
        if isinstance(data, dict):
            result_map = (data.get("formatted_results") or {}).get("result")
        else:
            result_map = None
        if not isinstance(result_map, dict):
            return [
                SkipRecord(
                    source_entry=source_entry,
                    source=self.template_id,
                    reason="malformed_record",
                    detail="missing formatted_results.result",
                )
            ]

        out: list[dict[str, Any] | SkipRecord] = []
        for uid in result_map.get("uids") or []:
            rec = result_map.get(uid) or {}
            skip = self._convert_variant(rec, source_entry, source_time)
            out.append(skip)

        return out

    def _convert_variant(
        self,
        rec: dict[str, Any],
        source_entry: str,
        source_time: str,
    ) -> dict[str, Any] | SkipRecord:
        # ── disease: MONDO xref first, resolver fallback ────────────────
        disease_ids: list[str] = []
        unmapped: list[str] = []
        gc = rec.get("germline_classification") or {}
        for trait in gc.get("trait_set") or []:
            xrefs = trait.get("trait_xrefs") or []
            curie = None
            for x in xrefs:
                if isinstance(x, dict) and x.get("db_source") == "MONDO" and x.get("db_id"):
                    curie = _canon_mondo(str(x["db_id"]))
                    break
            if curie is None:
                tname = str(trait.get("trait_name") or "").strip()
                curie = resolve_disease(tname) if tname else None
            if curie:
                if curie not in disease_ids:
                    disease_ids.append(curie)
            else:
                unmapped.append(str(trait.get("trait_name") or ""))
        if not disease_ids:
            return SkipRecord(
                source_entry=source_entry,
                source=self.template_id,
                reason="unmapped_disease",
                detail=";".join(unmapped[:5]) or "no trait information",
            )

        # ── gene entity (label only — no invented CURIE) ─────────────────
        genes = rec.get("genes") or []
        gene_symbol = ""
        if isinstance(genes, list) and genes:
            if isinstance(genes[0], dict):
                gene_symbol = str(genes[0].get("symbol") or "")
            else:
                gene_symbol = str(genes[0])

        # ── variant description ──────────────────────────────────────────
        variant_title = str(rec.get("title") or "").strip()
        if not variant_title:
            variant_title = str(rec.get("accession") or "variant")

        gene_entities = []
        if gene_symbol:
            gene_entities.append({"entity_id": None, "entity_label": gene_symbol})

        raw_evidence = {
            "biological_level": "epi",
            "evidence_design": "observational_association",
            "xy": {"count": 1},
            "x": {
                "type": "gene",
                "value": gene_entities or [{"entity_id": None, "entity_label": "variant"}],
            },
            "y": {
                "type": variant_title[:200],
                "category": "clinic_efficacy_primary",
                "value": [1.0],
                "to_basis": "absolute",
                "direction": "harmful",
            },
            "bg": {"disease_id": disease_ids, "drugs": [], "genes": gene_entities},
            "clinical_design": {"comparator_type": "no_treatment", "population": "adults"},
            "source_entry": source_entry,
            "source_time": source_time,
        }
        return raw_evidence


def _canon_mondo(curie: str) -> str:
    prefix, _, accession = curie.partition(":")
    return f"{prefix.lower()}:{accession}"
