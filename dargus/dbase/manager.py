"""DBaseManager v0.17.0 — sole write/read gate with three-axis evidence dict API."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from dargus.dbase.dbase import DBase
from dargus.dbase.validate import compute_evidence_id, validate_evidence

if TYPE_CHECKING:
    from dargus.dbase.nlp import DBaseNLP
    from dargus.models.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class DuplicateReviewRequest:
    """Soft-flag when semantic similarity >= threshold (v0.15.5 three-axis fields)."""

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
    """Single read/write interface to D-Base (v0.17.0 three-axis)."""

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

    # ── read (§8.2) ─────────────────────────────────────────────────────────

    def read_records(
        self,
        x_entity: str | None = None,
        disease_id: str | None = None,
        y_type: str | None = None,
        y_category: str | None = None,
        level: str | None = None,
        evidence_design: str | None = None,
        *,
        drug_id: str | None = None,
        template_id: str | None = None,
        readout_type: str | None = None,
        intervention_id: str | None = None,
    ) -> list[dict]:
        """Read evidence records matching three-axis filters.

        Transitional compat aliases (deprecation warning):
          drug_id → x_entity
          template_id → ignored
          readout_type → y_type
          intervention_id → x_entity
        """
        if drug_id is not None and x_entity is None:
            warnings.warn("'drug_id' is deprecated, use 'x_entity' instead", DeprecationWarning)
            x_entity = drug_id
        if template_id is not None:
            warnings.warn("'template_id' is deprecated and ignored", DeprecationWarning)
        if readout_type is not None and y_type is None:
            warnings.warn("'readout_type' is deprecated, use 'y_type' instead", DeprecationWarning)
            y_type = readout_type
        if intervention_id is not None and x_entity is None:
            warnings.warn(
                "'intervention_id' is deprecated, use 'x_entity' instead", DeprecationWarning
            )
            x_entity = intervention_id

        return self.dbase.query_parquet(
            readout_type=y_type,
            readout_category=y_category,
            intervention_id=x_entity,
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

    # ── write (§8.1) ────────────────────────────────────────────────────────

    def write_record(self, record: dict, dedup: bool = True) -> bool | DuplicateReviewRequest:
        """Write one three-axis evidence dict to D-Base.

        1. validate(§6) → hard reject aborts
        2. compute evidence_id(§5) → collision = duplicate (return False)
        3. semantic check → DuplicateReviewRequest if similar
        4. append shard → mark view stale
        """
        result = validate_evidence(record)
        if not result.ok:
            raise ValueError(f"Validation failed: {'; '.join(result.hard_errors)}")

        if result.soft_warnings:
            record["needs_curation"] = True

        eid = compute_evidence_id(record)
        record["evidence_id"] = eid

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
            from dargus.dbase.nlp import DBaseNLP

            text = DBaseNLP.record_to_text(record)
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
        except Exception:
            pass
        return None

    # ── build_evidence (§8.3) ───────────────────────────────────────────────

    def build_evidence(
        self,
        raw_input: dict[str, Any],
        source_metadata: dict[str, Any],
        biological_level: str | None = None,
    ) -> dict:
        """Assemble raw fields into a three-axis evidence dict. Does NOT persist."""
        evidence: dict[str, Any] = {}

        # top-level identity
        if biological_level:
            evidence["biological_level"] = biological_level
        if "biological_level" in raw_input:
            evidence["biological_level"] = raw_input["biological_level"]

        evidence["evidence_design"] = raw_input.get("evidence_design", "descriptive")

        # xy axis
        xy_count = raw_input.get("xy", {}).get("count", 0)
        evidence.setdefault("xy", {})["count"] = xy_count

        # x axis
        x_type = raw_input.get("x", {}).get("type", "drug")
        x_unit = raw_input.get("x", {}).get("unit")
        x_vals = raw_input.get("x", {}).get("value", [])
        evidence["x"] = {"type": x_type, "unit": x_unit, "value": x_vals}

        # y axis
        y_type = raw_input.get("y", {}).get("type") or raw_input.get("readout_type", "")
        y_cat = raw_input.get("y", {}).get("category") or raw_input.get("readout_category", "")
        y_unit = raw_input.get("y", {}).get("unit") or raw_input.get("readout_unit")
        y_basis = raw_input.get("y", {}).get("basis", "absolute")
        y_vals = (raw_input.get("y") or {}).get("value")
        if y_vals is None:
            rv = raw_input.get("readout_value")
            if rv is not None:
                y_vals = [rv] if not isinstance(rv, list) else rv
            else:
                y_vals = []
        y_ci95 = raw_input.get("y", {}).get("ci95") or []
        y_n = raw_input.get("y", {}).get("n_total") or []
        y_pval = raw_input.get("y", {}).get("p_value") or []
        y_dir = raw_input.get("y", {}).get("direction") or raw_input.get("readout_direction")
        y_eff = raw_input.get("y", {}).get("effect")

        # backward compat: build ci95 from flat fields
        if not y_ci95:
            ci_low = raw_input.get("ci95_lower")
            ci_up = raw_input.get("ci95_upper")
            if ci_low is not None and ci_up is not None:
                y_ci95 = [{"lower": ci_low, "upper": ci_up}]

        # backward compat: build n_total from flat n_total
        if not y_n and raw_input.get("n_total") is not None:
            n = raw_input["n_total"]
            y_n = [n] if not isinstance(n, list) else n

        # backward compat: build p_value from flat p_value
        if not y_pval and raw_input.get("p_value") is not None:
            pv = raw_input["p_value"]
            y_pval = [pv] if not isinstance(pv, (list, tuple)) else list(pv)

        # Backward compat: drug_id / drug → store in bg.drugs (descriptive records)
        drug_id = raw_input.get("drug_id") or raw_input.get("drug")
        if drug_id:
            entity_id = drug_id if ":" in str(drug_id) else f"chembl:{drug_id}"
            _bg_drugs_from_compat = [{"entity_id": entity_id}]
        else:
            _bg_drugs_from_compat = []

        # Assemble y
        evidence["y"] = {"type": y_type, "category": y_cat}
        if y_unit:
            evidence["y"]["unit"] = y_unit
        evidence["y"]["basis"] = y_basis
        evidence["y"]["value"] = y_vals
        if y_ci95:
            evidence["y"]["ci95"] = y_ci95
        if y_n:
            evidence["y"]["n_total"] = y_n
        if y_pval:
            evidence["y"]["p_value"] = y_pval
        if y_dir:
            evidence["y"]["direction"] = y_dir
        if y_eff:
            evidence["y"]["effect"] = y_eff

        # bg axis
        disease_id = raw_input.get("bg", {}).get("disease_id") or raw_input.get("disease_id")
        if isinstance(disease_id, str):
            disease_id = [disease_id]
        bg_drugs = (raw_input.get("bg") or {}).get("drugs", [])
        if _bg_drugs_from_compat:
            bg_drugs = _bg_drugs_from_compat + bg_drugs
        bg_genes = raw_input.get("bg", {}).get("genes", [])
        bg_model = raw_input.get("bg", {}).get("model")
        evidence["bg"] = {
            "disease_id": disease_id or [],
            "drugs": bg_drugs,
            "genes": bg_genes,
            "model": bg_model,
        }

        # sample identity — flattened from old experimental_context
        ec = raw_input.get("experimental_context") or {}
        for key in ("cell_line_id", "model_organism", "strain", "sex"):
            val = raw_input.get(key) or ec.get(key)
            if val is not None:
                evidence[key] = val

        # tissue / cell_type — flattened from old sample
        sample = raw_input.get("sample") or {}
        for key in ("tissue", "cell_type"):
            val = raw_input.get(key) or sample.get(key)
            if val is not None:
                evidence[key] = val

        # assay/exposure — flattened from old platform/exposure
        platform = raw_input.get("platform") or {}
        _ap = raw_input.get("assay_platform") or platform.get("assay_platform")
        evidence["assay_platform"] = _ap
        _ep = raw_input.get("exvivo_platform") or platform.get("exvivo_platform")
        evidence["exvivo_platform"] = _ep

        _exp_keys = (
            "exposure_dose_value",
            "exposure_dose_unit",
            "exposure_duration_value",
            "exposure_duration_unit",
        )
        for key in _exp_keys:
            val = raw_input.get(key)
            if val is not None:
                evidence[key] = val
        exposure = raw_input.get("exposure") or {}
        dose = exposure.get("dose") or {}
        if "exposure_dose_value" not in evidence and dose.get("value") is not None:
            evidence["exposure_dose_value"] = dose["value"]
            evidence["exposure_dose_unit"] = dose.get("unit")
        dur = exposure.get("duration") or {}
        if "exposure_duration_value" not in evidence and dur.get("value") is not None:
            evidence["exposure_duration_value"] = dur["value"]
            evidence["exposure_duration_unit"] = dur.get("unit")

        # clinical_design
        if "clinical_design" in raw_input:
            evidence["clinical_design"] = raw_input["clinical_design"]

        # sources
        evidence["sources"] = source_metadata.get("sources", [])
        if not evidence["sources"] and source_metadata:
            evidence["sources"] = [
                {
                    "rank": 1,
                    "type": source_metadata.get("type", "url"),
                    "id": source_metadata.get("id", ""),
                }
            ]

        # metadata
        for key in (
            "phenotypes",
            "is_primary_endpoint",
            "p_value_adjusted",
            "llm_summary",
            "status",
            "superseded_by",
            "revision",
            "legacy_hash",
            "experiment_group_id",
        ):
            if key in raw_input and raw_input[key] is not None:
                evidence[key] = raw_input[key]

        # Validate and stamp
        result = validate_evidence(evidence)
        if not result.ok:
            raise ValueError(f"build_evidence validation failed: {'; '.join(result.hard_errors)}")
        if result.soft_warnings:
            evidence["needs_curation"] = True

        evidence["evidence_id"] = compute_evidence_id(evidence)
        return evidence

    # ── semantic (§8.5) ─────────────────────────────────────────────────────

    def read_records_semantic(
        self,
        query_text: str,
        top_k: int = 10,
        x_entity: str | None = None,
        disease_id: str | None = None,
        y_type: str | None = None,
        *,
        intervention_id: str | None = None,
        readout_type: str | None = None,
    ) -> list[tuple[dict, float]]:
        """Semantic search over evidence records."""
        _x_entity = x_entity or intervention_id
        _y_type = y_type or readout_type

        try:
            from dargus.dbase.nlp import DBaseNLP

            nlp = self.nlp
            candidates = self.read_records(
                x_entity=_x_entity,
                disease_id=disease_id,
                y_type=_y_type,
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

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all records from D-Base."""
        self.dbase.clear()

    def _record_field(self, record: dict, field_name: str) -> Any:
        """Extract a field from a three-axis evidence dict."""
        if field_name in record:
            return record[field_name]
        # resolve from three-axis structure
        if field_name == "drug_id" or field_name == "x_entity":
            xv = record.get("x", {}).get("value") or []
            if xv:
                return xv[0].get("entity_id")
        if field_name == "endpoint" or field_name == "y_type":
            return record.get("y", {}).get("type")
        if field_name == "fold_change" or field_name == "y_value":
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
