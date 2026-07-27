"""DBaseManager v1.0.0 — sole write/read gate with three-axis evidence dict API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dargus.dbase.dbase import DBase
from dargus.dbase.validate import compute_evidence_id, validate_evidence

if TYPE_CHECKING:
    from dargus.dbase.nlp import DBaseNLP
    from dargus.models.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class DuplicateReviewRequest:
    """Soft-flag when semantic similarity >= threshold."""

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
    """Single read/write interface to D-Base (v1.0.0 three-axis, 50-field)."""

    def __init__(self, dbase: DBase, embedding_model: EmbeddingModel | None = None) -> None:
        self.dbase = dbase
        self._embedding_model = embedding_model
        self._nlp: DBaseNLP | None = None

    # ── nlp (lazy) ──────────────────────────────────────────────────────────

    @property
    def nlp(self) -> DBaseNLP:
        """Lazily-initialised DBaseNLP instance, wired to the manager's EmbeddingModel."""
        if self._nlp is None:
            from dargus.dbase.nlp import DBaseNLP

            self._nlp = DBaseNLP(embedding_model=self._embedding_model)
        return self._nlp

    # ── read ────────────────────────────────────────────────────────────────

    def read_records(
        self,
        x_entity: str | None = None,
        disease_id: str | None = None,
        y_type: str | None = None,
        y_category: str | None = None,
        level: str | None = None,
        evidence_design: str | None = None,
    ) -> list[dict]:
        """Read evidence records matching three-axis filters."""
        return self.dbase.query_parquet(
            y_type=y_type,
            y_category=y_category,
            x_entity=x_entity,
            disease_id=disease_id,
            biological_level=level,
            evidence_design=evidence_design,
        )

    def read_record(self, evidence_id: str) -> dict | None:
        """Read a single evidence record by content-addressed id."""
        for record in self.dbase.read_shards():
            if record.get("evidence_id") == evidence_id:
                return record
        return None

    # ── write ───────────────────────────────────────────────────────────────

    def write_record(self, record: dict, dedup: bool = True) -> bool | DuplicateReviewRequest:
        """Write one three-axis evidence dict to D-Base.

        1. validate → hard reject aborts
        2. compute evidence_id → collision = duplicate (return False)
        3. generate embedding (optional; best-effort)
        4. semantic check → DuplicateReviewRequest if similar
        5. append shard → mark view stale
        """
        result = validate_evidence(record)
        if not result.ok:
            raise ValueError(f"Validation failed: {'; '.join(result.hard_errors)}")

        if result.soft_warnings:
            record["needs_curation"] = True

        eid = compute_evidence_id(record)
        record["evidence_id"] = eid

        # ── generate + store embedding (best-effort) ──────────────────────
        try:
            text = self.nlp.record_to_text(record)
            emb = self.nlp.embed_text(text)
            record["embedding"] = emb.tolist()
        except Exception:
            pass  # embedding is optional; record still writable

        if dedup:
            if self.dbase.evidence_id_exists(eid):
                return False

            dup = self._semantic_check(record)
            if dup:
                return dup

        self.dbase.append_shard(record)
        self.dbase.mark_view_stale()
        return True

    def _semantic_check(self, record: dict) -> DuplicateReviewRequest | None:
        try:
            text = self.nlp.record_to_text(record)
            similar = self.read_records_semantic(
                text,
                top_k=5,
                x_entity=_primary_x_entity(record),
                disease_id=_primary_disease_id(record),
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        return None

    # ── build_evidence ──────────────────────────────────────────────────────

    def build_evidence(
        self,
        raw_input: dict[str, Any],
        source_metadata: dict[str, Any],
        biological_level: str | None = None,
    ) -> dict:
        """Assemble raw fields into a v1.0.0 three-axis evidence dict. Does NOT persist.

        source_metadata keys:
          sources       — full [{rank, type, name}] list, or
          type / name   — single rank-1 source shorthand
          entry / time  — fallbacks for raw_input["source_entry"] / ["source_time"]
        """
        evidence: dict[str, Any] = {}

        if biological_level:
            evidence["biological_level"] = biological_level
        if "biological_level" in raw_input:
            evidence["biological_level"] = raw_input["biological_level"]

        evidence["evidence_design"] = raw_input.get("evidence_design", "descriptive")

        # xy axis (default: single descriptive data point)
        xy_count = raw_input.get("xy", {}).get("count", 1)
        evidence.setdefault("xy", {})["count"] = xy_count

        # x axis
        x_raw = raw_input.get("x", {})
        x_type = x_raw.get("type", "drug")
        x_unit = x_raw.get("unit")
        x_vals = x_raw.get("value")
        if x_vals is None:
            # shorthand: drug_id / drug → single drug data point
            drug_id = raw_input.get("drug_id") or raw_input.get("drug")
            if drug_id:
                entity_id = str(drug_id) if ":" in str(drug_id) else f"chembl:{drug_id}"
                label = raw_input.get("drug_label") or str(drug_id)
                x_vals = [{"entity_id": entity_id, "entity_label": label}]
                x_type = "drug"
            else:
                x_vals = []
        evidence["x"] = {"type": x_type, "value": x_vals}
        if x_unit:
            evidence["x"]["unit"] = x_unit

        # y axis
        y_raw = raw_input.get("y", {})
        y_type = y_raw.get("type") or raw_input.get("readout_type", "")
        y_cat = y_raw.get("category") or raw_input.get("readout_category", "")
        y_unit = y_raw.get("unit") or raw_input.get("readout_unit")
        y_to_basis = y_raw.get("to_basis")
        y_vals = y_raw.get("value")
        if y_vals is None:
            rv = raw_input.get("readout_value")
            if rv is not None:
                y_vals = [rv] if not isinstance(rv, list) else rv
            else:
                y_vals = []
        y_dispersion = y_raw.get("dispersion") or []
        # shorthand: flat ci95 lower/upper → y.dispersion entries of type CI95
        if not y_dispersion:
            ci_low = raw_input.get("ci95_lower")
            ci_up = raw_input.get("ci95_upper")
            if ci_low is not None and ci_up is not None:
                y_dispersion = [{"type": "CI95", "value": [ci_low, ci_up]}]
        y_n = y_raw.get("n_total") or []
        if not y_n and raw_input.get("n_total") is not None:
            n = raw_input["n_total"]
            y_n = [n] if not isinstance(n, list) else n
        y_pval = y_raw.get("p_value") or []
        if not y_pval and raw_input.get("p_value") is not None:
            pv = raw_input["p_value"]
            y_pval = [pv] if not isinstance(pv, (list, tuple)) else list(pv)
        y_events = y_raw.get("events") or []
        y_dir = y_raw.get("direction") or raw_input.get("readout_direction")
        y_eff = y_raw.get("effect")
        y_assay = y_raw.get("assay") or raw_input.get("assay_platform")

        evidence["y"] = {"type": y_type, "category": y_cat, "value": y_vals}
        if y_unit:
            evidence["y"]["unit"] = y_unit
        if y_to_basis:
            evidence["y"]["to_basis"] = y_to_basis
        if y_dispersion:
            evidence["y"]["dispersion"] = y_dispersion
        if y_n:
            evidence["y"]["n_total"] = y_n
        if y_pval:
            evidence["y"]["p_value"] = y_pval
        if y_events:
            evidence["y"]["events"] = y_events
        if y_dir:
            evidence["y"]["direction"] = y_dir
        if y_eff:
            evidence["y"]["effect"] = y_eff
        if y_assay:
            evidence["y"]["assay"] = y_assay

        # bg axis
        bg_raw = raw_input.get("bg", {})
        disease_id = bg_raw.get("disease_id") or raw_input.get("disease_id") or []
        if isinstance(disease_id, str):
            disease_id = [disease_id]
        bg_drugs = bg_raw.get("drugs", [])
        bg_genes = bg_raw.get("genes", [])
        bg_model = bg_raw.get("model")
        evidence["bg"] = {
            "disease_id": disease_id,
            "drugs": bg_drugs,
            "genes": bg_genes,
        }
        if bg_model:
            evidence["bg"]["model"] = bg_model
        for key in ("dose_value", "dose_unit", "duration_value", "duration_unit", "phenotype"):
            if key in bg_raw and bg_raw[key] is not None:
                evidence["bg"][key] = bg_raw[key]
        # exposure shorthand: raw_input["exposure"] = {"dose": {value, unit}, "duration": {...}}
        exposure = raw_input.get("exposure") or {}
        dose = exposure.get("dose") or {}
        if "dose_value" not in evidence["bg"] and dose.get("value") is not None:
            evidence["bg"]["dose_value"] = dose["value"]
            evidence["bg"]["dose_unit"] = dose.get("unit")
        duration = exposure.get("duration") or {}
        if "duration_value" not in evidence["bg"] and duration.get("value") is not None:
            evidence["bg"]["duration_value"] = duration["value"]
            evidence["bg"]["duration_unit"] = duration.get("unit")

        # sample fields
        ec = raw_input.get("experimental_context") or {}
        for key in ("cell_line_id", "model_organism", "strain", "sex", "age"):
            val = raw_input.get(key) or ec.get(key)
            if val is not None:
                evidence[key] = val
        sample = raw_input.get("sample") or {}
        for key in ("tissue", "cell_type"):
            val = raw_input.get(key) or sample.get(key)
            if val is not None:
                evidence[key] = val
        platform = raw_input.get("platform") or {}
        _ep = raw_input.get("exvivo_platform") or platform.get("exvivo_platform")
        if _ep is not None:
            evidence["exvivo_platform"] = _ep

        # clinical_design
        if "clinical_design" in raw_input:
            evidence["clinical_design"] = raw_input["clinical_design"]

        # provenance
        sources = source_metadata.get("sources")
        if not sources:
            stype = source_metadata.get("type", "database")
            sname = source_metadata.get("name") or source_metadata.get("id", "")
            sources = [{"rank": 1, "type": stype, "name": sname}]
        evidence["sources"] = sources
        entry = raw_input.get("source_entry") or source_metadata.get("entry", "")
        time_ = raw_input.get("source_time") or source_metadata.get("time", "")
        if entry:
            evidence["source_entry"] = entry
        if time_:
            evidence["source_time"] = time_

        # metadata
        for key in ("related_evidence_id", "is_primary_endpoint", "p_value_adjusted"):
            if key in raw_input and raw_input[key] is not None:
                evidence[key] = raw_input[key]

        result = validate_evidence(evidence)
        if not result.ok:
            raise ValueError(f"build_evidence validation failed: {'; '.join(result.hard_errors)}")
        if result.soft_warnings:
            evidence["needs_curation"] = True

        evidence["evidence_id"] = compute_evidence_id(evidence)
        return evidence

    # ── semantic ────────────────────────────────────────────────────────────

    def read_records_semantic(
        self,
        query_text: str,
        top_k: int = 10,
        x_entity: str | None = None,
        disease_id: str | None = None,
        y_type: str | None = None,
    ) -> list[tuple[dict, float]]:
        """Semantic search over evidence records."""
        try:
            nlp = self.nlp
            candidates = self.read_records(
                x_entity=x_entity,
                disease_id=disease_id,
                y_type=y_type,
            )
            scored: list[tuple[dict, float]] = []
            for record in candidates:
                text = nlp.record_to_text(record)
                score = nlp.similarity(query_text, text)
                scored.append((record, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all records from D-Base."""
        self.dbase.clear()

    def _record_field(self, record: dict, field_name: str) -> Any:
        """Extract a field from a three-axis evidence dict."""
        if field_name in record:
            return record[field_name]
        if field_name in ("drug_id", "x_entity"):
            xv = record.get("x", {}).get("value") or []
            if xv:
                return xv[0].get("entity_id")
        if field_name in ("endpoint", "y_type"):
            return record.get("y", {}).get("type")
        if field_name in ("fold_change", "y_value"):
            yv = record.get("y", {}).get("value") or []
            return yv[0] if yv else None
        if field_name == "disease_id":
            dids = record.get("bg", {}).get("disease_id") or []
            return dids[0] if dids else None
        return None


def _primary_x_entity(record: dict) -> str | None:
    xv = record.get("x", {}).get("value") or []
    if xv:
        return xv[0].get("entity_id") or xv[0].get("entity_label")
    return None


def _primary_disease_id(record: dict) -> str | None:
    dids = record.get("bg", {}).get("disease_id") or []
    return dids[0] if dids else None
