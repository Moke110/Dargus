"""DBaseManager v0.15.0 — sole access gate to D-Base (evidence dict API)."""

from __future__ import annotations

from typing import Any

from dargus.dbase.dbase import DBase
from dargus.dbase.validate import compute_evidence_id, validate_evidence


class DuplicateReviewRequest:
    """Soft-flag result when semantic similarity >= threshold."""

    def __init__(
        self,
        incoming_raw: dict[str, Any],
        incoming_evidence: dict[str, Any],
        candidate_evidence: dict[str, Any],
        similarity_score: float,
        candidate_evidence_id: str,
    ) -> None:
        self.incoming_raw = incoming_raw
        self.incoming_evidence = incoming_evidence
        self.candidate_evidence = candidate_evidence
        self.similarity_score = similarity_score
        self.candidate_evidence_id = candidate_evidence_id


class DBaseManager:
    """Single read/write interface to D-Base (v0.15.0)."""

    def __init__(self, dbase: DBase) -> None:
        self.dbase = dbase

    # ── read ──────────────────────────────────────────────────────────────

    def read_records(
        self,
        readout_type: str | None = None,
        readout_category: str | None = None,
        intervention_id: str | None = None,
        disease_id: str | None = None,
        biological_level: str | None = None,
        evidence_design: str | None = None,
        *,
        drug_id: str | None = None,
        template_id: str | None = None,
    ) -> list[dict]:
        """Read evidence records matching filters.

        Compat aliases: drug_id → intervention_id, template_id → readout_type.
        """
        iid = intervention_id or drug_id
        rt = readout_type or template_id
        return self.dbase.query_parquet(
            readout_type=rt,
            readout_category=readout_category,
            intervention_id=iid,
            disease_id=disease_id,
            biological_level=biological_level,
        )

    def read_record(self, evidence_id: str) -> dict | None:
        """Read a single evidence record by content-addressed id."""
        for record in self.dbase.read_shards():
            if record.get("evidence_id") == evidence_id:
                return record
        return None

    # ── write ─────────────────────────────────────────────────────────────

    def write_record(self, record: dict, dedup: bool = True) -> bool | DuplicateReviewRequest:
        """Write one evidence dict to D-Base.

        When dedup=True:
        1. Validate → hard reject aborts
        2. Compute evidence_id → collision = duplicate (return False)
        3. Semantic check → DuplicateReviewRequest if similar
        4. Append shard → mark view stale
        """
        # Validate
        result = validate_evidence(record)
        if not result.ok:
            raise ValueError(f"Validation failed: {'; '.join(result.hard_errors)}")

        # Apply soft warnings
        if result.soft_warnings:
            record["needs_curation"] = True

        # Compute evidence_id
        eid = compute_evidence_id(record)
        record["evidence_id"] = eid
        record.setdefault("schema_version", "v0.15.0")

        if dedup:
            # Exact dedup
            if self.dbase.evidence_id_exists(eid):
                return False

            # Semantic dedup
            dup = self._semantic_check(record)
            if dup:
                return dup

        # Append + mark view stale
        self.dbase.append_shard(record)
        self.dbase.mark_view_stale()
        return True

    def _semantic_check(self, record: dict) -> DuplicateReviewRequest | None:
        """Check for semantically similar evidence."""
        try:
            from dargus.dbase.nlp import DBaseNLP

            nlp = DBaseNLP()
            text = DBaseNLP.record_to_text(record)
            similar = self.read_records_semantic(
                text,
                top_k=5,
                intervention_id=_primary_intervention_id(record),
                disease_id=record.get("disease_id"),
            )
            for candidate, score in similar:
                if score >= 0.85:
                    return DuplicateReviewRequest(
                        incoming_raw=record,
                        incoming_evidence=record,
                        candidate_evidence=candidate,
                        similarity_score=score,
                        candidate_evidence_id=candidate.get("evidence_id", ""),
                    )
        except Exception:
            pass
        return None

    # ── ingestion ──────────────────────────────────────────────────────────

    def build_evidence(
        self,
        raw_input: dict[str, Any],
        source_metadata: dict[str, Any],
        biological_level: str | None = None,
    ) -> dict:
        """Assemble extracted raw fields into a validated evidence dict.

        Replaces the old fill_template(). Does NOT persist — always goes through
        write_record() afterwards.
        """
        evidence: dict[str, Any] = {}

        # Identity
        if biological_level:
            evidence["biological_level"] = biological_level
        if "biological_level" in raw_input:
            evidence["biological_level"] = raw_input["biological_level"]

        evidence["disease_id"] = raw_input.get("disease_id")
        evidence["readout_type"] = raw_input.get("readout_type")
        evidence["readout_category"] = raw_input.get("readout_category")
        evidence["evidence_design"] = raw_input.get("evidence_design", "two_arm_comparison")

        # Intervention — from raw_input
        interventions = raw_input.get("interventions", [])
        if not interventions:
            drug_id = raw_input.get("drug_id") or raw_input.get("drug")
            if drug_id:
                entity_id = drug_id if ":" in str(drug_id) else f"chembl:{drug_id}"
                interventions = [
                    {
                        "role": "primary",
                        "entity_type": "small_molecule",
                        "entity_id": entity_id,
                    }
                ]
            target_id = raw_input.get("target_id") or raw_input.get("target")
            if target_id:
                target_eid = target_id if ":" in str(target_id) else f"uniprot:{target_id}"
                interventions.append(
                    {
                        "role": "comparator_agent",
                        "entity_type": "gene",
                        "entity_id": target_eid,
                        "alteration": raw_input.get("target_alteration")
                        or raw_input.get("alteration"),
                    }
                )
        evidence["interventions"] = interventions

        # Readout values
        for key in (
            "readout_value",
            "readout_unit",
            "n_total",
            "p_value",
            "statistical_test",
            "effect_direction",
            "is_qualitative",
            "readout_direction",
        ):
            if key in raw_input and raw_input[key] is not None:
                evidence[key] = raw_input[key]

        if "ci95_lower" in raw_input and "ci95_upper" in raw_input:
            evidence["readout_ci95"] = {
                "lower": raw_input["ci95_lower"],
                "upper": raw_input["ci95_upper"],
            }

        # Sources
        evidence["sources"] = source_metadata.get("sources", [])
        if not evidence["sources"] and source_metadata:
            evidence["sources"] = [
                {
                    "rank": 1,
                    "type": source_metadata.get("type", "url"),
                    "id": source_metadata.get("id", ""),
                }
            ]

        # Validate and stamp
        result = validate_evidence(evidence)
        if not result.ok:
            raise ValueError(f"build_evidence validation failed: {'; '.join(result.hard_errors)}")
        if result.soft_warnings:
            evidence["needs_curation"] = True

        evidence["evidence_id"] = compute_evidence_id(evidence)
        evidence["schema_version"] = "v0.15.0"
        return evidence

    # ── semantic ───────────────────────────────────────────────────────────

    def read_records_semantic(
        self,
        query_text: str,
        top_k: int = 10,
        readout_type: str | None = None,
        intervention_id: str | None = None,
        disease_id: str | None = None,
    ) -> list[tuple[dict, float]]:
        """Semantic search over evidence records."""
        try:
            from dargus.dbase.nlp import DBaseNLP

            nlp = DBaseNLP()
            candidates = self.read_records(
                readout_type=readout_type,
                intervention_id=intervention_id,
                disease_id=disease_id,
            )
            scored: list[tuple[dict, float]] = []
            for record in candidates:
                text = DBaseNLP.record_to_text(record)
                score = nlp.similarity(query_text, text)
                scored.append((record, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception:
            return []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all records from D-Base."""
        self.dbase.clear()


def _primary_intervention_id(record: dict) -> str | None:
    for iv in record.get("interventions", []):
        if iv.get("role") == "primary":
            return iv.get("entity_id")
    return None
