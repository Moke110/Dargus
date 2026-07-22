"""D-Base v0.15.0 reference validator.

Mirrors the write path: hard reject = cannot write; soft warning = write + needs_curation.

Usage:
    from dargus.dbase.validate import validate_evidence
    result = validate_evidence(evidence_dict)
    if result.hard_errors:
        raise ValidationError(result.hard_errors)
    if result.soft_warnings:
        evidence["needs_curation"] = True
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# ── vocabulary ────────────────────────────────────────────────────────────────

BIOLOGICAL_LEVELS = frozenset(
    {
        "molecular",
        "molecular-sim",
        "cellular",
        "cellular-sim",
        "exvivo",
        "exvivo-sim",
        "animal",
        "animal-sim",
        "rct",
        "epi",
        "rct-sim",
    }
)

CLINICAL_LEVELS = frozenset({"rct", "epi", "rct-sim"})
SIM_LEVELS = frozenset({level for level in BIOLOGICAL_LEVELS if level.endswith("-sim")})
NON_CLINICAL_LEVELS = BIOLOGICAL_LEVELS - CLINICAL_LEVELS

READOUT_CATEGORIES = frozenset(
    {
        "clinic_efficacy_primary",
        "clinic_efficacy_secondary",
        "clinic_efficacy_exploratory",
        "clinic_toxicity_primary",
        "clinic_toxicity_secondary",
        "clinic_toxicity_exploratory",
        "binding",
        "pk_adme",
        "prot_exp",
        "rna_exp",
        "viability",
        "apoptosis",
        "proliferation",
        "migration",
        "invasion",
        "autophagy",
        "differentiation",
        "phosphorylation",
        "localization",
        "metabolism",
        "oxidative_stress",
        "behavioral",
        "other",
    }
)

EVIDENCE_DESIGNS = frozenset(
    {
        "two_arm_comparison",
        "single_arm",
        "dose_escalation",
        "dose_response_curve",
        "observational_association",
        "continuous_trajectory",
        "descriptive",
    }
)

INTERVENTION_ROLES = frozenset(
    {
        "primary",
        "combination_partner",
        "background_therapy",
        "comparator_agent",
    }
)

ENTITY_TYPES = frozenset({"small_molecule", "biologic", "gene", "combination", "other"})

ALTERATIONS = frozenset({"KO", "OE", "KD", "GOF", "LOF", "DN", "CRISPRi", "CRISPRa"})

EFFECT_TYPES = frozenset(
    {
        "cohens_d",
        "hedges_g",
        "smd",
        "raw_mean_diff",
        "ratio",
        "odds_ratio",
        "risk_ratio",
        "hazard_ratio",
        "log_or",
        "log_rr",
        "log_hr",
        "correlation",
        "other",
    }
)

LOG_EFFECT_TYPES = frozenset({"log_or", "log_rr", "log_hr"})

SOURCE_TYPES = frozenset({"doi", "pmid", "pmcid", "db_accession", "file_path", "url"})

# CURIE prefix → accession regex (hard-validated prefixes)
CURIE_PATTERNS: dict[str, re.Pattern] = {
    "chembl": re.compile(r"^CHEMBL\d+$"),
    "uniprot": re.compile(
        r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$"
    ),
    "mondo": re.compile(r"^\d{7}$"),
    "doid": re.compile(r"^\d+$"),
    "hp": re.compile(r"^\d{7}$"),
    "meddra": re.compile(r"^\d{8}$"),
    "uberon": re.compile(r"^\d{7}$"),
    "cl": re.compile(r"^\d{7}$"),
    "cellosaurus": re.compile(r"^CVCL_[A-Z0-9]{4}$"),
    "NCBITaxon": re.compile(r"^\d+$"),
    "clinicaltrials": re.compile(r"^NCT\d{8}$"),
}

FALLBACK_PREFIXES = frozenset(
    {
        "drugbank",
        "rxnorm",
        "unii",
        "iuphar",
        "pubchem.compound",
        "complexportal",
        "refseq",
        "genbank",
        "insdc",
        "bto",
    }
)

ALL_PREFIXES = frozenset(CURIE_PATTERNS.keys()) | FALLBACK_PREFIXES

STRING_NULLS = frozenset({"null", "NA", "None", "nan", "N/A"})


@dataclass
class ValidationResult:
    hard_errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.hard_errors) == 0


# ── public API ─────────────────────────────────────────────────────────────────


def validate_evidence(evidence: dict) -> ValidationResult:
    """Run all validation rules. Returns ValidationResult."""
    result = ValidationResult()

    _rule_string_nulls(evidence, result)
    _rule_biological_level(evidence, result)
    _rule_sources(evidence, result)
    _rule_evidence_design(evidence, result)
    _rule_interventions(evidence, result)
    _rule_level_field_groups(evidence, result)
    _rule_simulation_provenance(evidence, result)
    _rule_readout(evidence, result)
    _rule_curies(evidence, result)

    return result


def compute_evidence_id(evidence: dict) -> str:
    """Compute content-addressed evidence_id (I5)."""
    identity = _build_identity(evidence)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── rules ──────────────────────────────────────────────────────────────────────


def _rule_string_nulls(evidence: dict, result: ValidationResult) -> None:
    """Rule 10: recursively scan for string nulls."""

    def _scan(obj, path=""):
        if isinstance(obj, str):
            if obj in STRING_NULLS:
                result.hard_errors.append(f"String null '{obj}' at {path or 'root'}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]")

    _scan(evidence)


def _rule_biological_level(evidence: dict, result: ValidationResult) -> None:
    """Rule 1: biological_level must be one of 11 values."""
    level = evidence.get("biological_level")
    if level not in BIOLOGICAL_LEVELS:
        result.hard_errors.append(f"biological_level '{level}' not in {sorted(BIOLOGICAL_LEVELS)}")
        return

    # I3: derive is_clinical (override user value)
    evidence["is_clinical"] = 1 if level in ("rct", "epi") else 0

    # Rules 1b/c
    ed = evidence.get("evidence_design")
    if ed and ed not in EVIDENCE_DESIGNS:
        result.hard_errors.append(f"evidence_design '{ed}' not in {sorted(EVIDENCE_DESIGNS)}")

    rc = evidence.get("readout_category")
    if rc and rc not in READOUT_CATEGORIES:
        result.hard_errors.append(f"readout_category '{rc}' not in {sorted(READOUT_CATEGORIES)}")


def _rule_sources(evidence: dict, result: ValidationResult) -> None:
    """Rule 15: sources non-empty, exactly one rank=1."""
    sources = evidence.get("sources", [])
    if not sources:
        result.hard_errors.append("sources is empty")
        return

    rank1_count = sum(1 for s in sources if s.get("rank") == 1)
    if rank1_count != 1:
        result.hard_errors.append(f"sources must have exactly one rank=1, got {rank1_count}")

    for i, s in enumerate(sources):
        stype = s.get("type", "")
        if stype not in SOURCE_TYPES:
            result.hard_errors.append(f"sources[{i}].type '{stype}' not in {sorted(SOURCE_TYPES)}")
        sid = s.get("id", "")
        _validate_source_id(stype, sid, i, result)


def _validate_source_id(stype: str, sid: str, idx: int, result: ValidationResult) -> None:
    """I7: soft-validate source id format."""
    if not sid:
        result.hard_errors.append(f"sources[{idx}].id is empty")
        return

    patterns = {
        "doi": re.compile(r"^10\.\S+$"),
        "pmid": re.compile(r"^\d+$"),
        "pmcid": re.compile(r"^PMC\d+$"),
        "db_accession": re.compile(r"^[a-z_]+:.+$"),
        "url": re.compile(r"^https?://"),
    }
    if stype in patterns and not patterns[stype].match(sid):
        result.soft_warnings.append(f"sources[{idx}].id '{sid}' format mismatch for type {stype}")


def _rule_evidence_design(evidence: dict, result: ValidationResult) -> None:
    """Rules 6, 7: series_point XOR exposure; CI validation; arm_stats bounds."""
    # I2: series_point and exposure are mutually exclusive
    has_series = bool(evidence.get("series_point"))
    has_exposure = bool(evidence.get("exposure"))
    if has_series and has_exposure:
        result.hard_errors.append("series_point and exposure are mutually exclusive (I2)")

    # I1: CI validation
    _validate_ci(evidence, "readout_ci95", evidence.get("readout_value"), result)
    effect = evidence.get("effect") or {}
    _validate_ci(evidence, "effect.ci95", effect.get("value"), result, prefix="effect.")

    # Rule 8: arm_stats bounds
    arm = evidence.get("arm_stats") or {}
    for role in ("intervention", "control"):
        arm_data = arm.get(role) or {}
        n = arm_data.get("n")
        if n is not None and n <= 0:
            result.hard_errors.append(f"arm_stats.{role}.n must be >0, got {n}")
        if arm.get("kind") == "continuous":
            sd = arm_data.get("sd")
            if sd is not None and sd < 0:
                result.hard_errors.append(f"arm_stats.{role}.sd must be >=0, got {sd}")
        if arm.get("kind") in ("binary", "time_to_event"):
            events = arm_data.get("events")
            if events is not None and n is not None and not (0 <= events <= n):
                result.hard_errors.append(f"arm_stats.{role}.events {events} not in [0, n={n}]")

    # Rule 9: p_value range
    pv = evidence.get("p_value")
    if pv is not None and not (0 <= pv <= 1):
        result.hard_errors.append(f"p_value {pv} not in [0,1]")

    # Rule 9b: effect.type ↔ scale
    if effect:
        etype = effect.get("type", "")
        escale = effect.get("scale", "linear")
        if etype in LOG_EFFECT_TYPES and escale != "log":
            result.hard_errors.append(f"effect.type '{etype}' requires scale='log', got '{escale}'")


def _validate_ci(
    evidence: dict, ci_key: str, point_value, result: ValidationResult, prefix: str = ""
) -> None:
    """I1: CI lower <= upper and point estimate within CI bounds."""
    ci = evidence.get(ci_key)
    if not ci:
        return
    lower = ci.get("lower")
    upper = ci.get("upper")
    if lower is None or upper is None:
        return
    if lower > upper:
        result.hard_errors.append(f"{prefix}{ci_key} lower {lower} > upper {upper}")
    if point_value is not None and not (lower <= point_value <= upper):
        result.hard_errors.append(f"{prefix}{ci_key} point {point_value} not in [{lower}, {upper}]")


def _rule_interventions(evidence: dict, result: ValidationResult) -> None:
    """Rules 3, 4, 14: interventions validation."""
    interventions = evidence.get("interventions", [])
    if not interventions:
        result.hard_errors.append("interventions is empty")
        return

    primary_count = sum(1 for i in interventions if i.get("role") == "primary")
    if primary_count != 1:
        result.hard_errors.append(
            f"interventions must have exactly one role=primary, got {primary_count}"
        )

    for idx, item in enumerate(interventions):
        role = item.get("role", "")
        etype = item.get("entity_type", "")
        eid = item.get("entity_id", "")
        elabel = item.get("entity_label", "")
        alt = item.get("alteration")

        if role not in INTERVENTION_ROLES:
            result.hard_errors.append(f"interventions[{idx}].role '{role}' invalid")
        if etype not in ENTITY_TYPES:
            result.hard_errors.append(f"interventions[{idx}].entity_type '{etype}' invalid")

        # I4: at least one of entity_id / entity_label non-empty
        if not eid and not elabel:
            result.hard_errors.append(
                f"interventions[{idx}]: entity_id and entity_label both empty (I4)"
            )

        # Rule 4 (I12): background_therapy + gene = hard reject
        if role == "background_therapy" and etype == "gene":
            result.hard_errors.append(
                f"interventions[{idx}]: role=background_therapy + entity_type=gene forbidden (I12)"
            )

        # Rule 14: alteration only on gene entities
        if alt is not None:
            if etype != "gene":
                result.hard_errors.append(
                    f"interventions[{idx}]: alteration '{alt}' on non-gene entity_type '{etype}'"
                )
            elif alt not in ALTERATIONS:
                result.hard_errors.append(
                    f"interventions[{idx}].alteration '{alt}' not in {sorted(ALTERATIONS)}"
                )


def _rule_level_field_groups(evidence: dict, result: ValidationResult) -> None:
    """Rule 5, 11, 12: level ↔ field group applicability."""
    level = evidence.get("biological_level", "")

    # Rule 5: clinical_design for clinical levels only; experimental_context for non-clinical
    if evidence.get("clinical_design") and level not in CLINICAL_LEVELS:
        result.hard_errors.append(
            f"clinical_design present but biological_level={level} (only for rct/epi/rct-sim)"
        )
    if evidence.get("experimental_context") and level in CLINICAL_LEVELS:
        result.hard_errors.append(
            f"experimental_context present but biological_level={level} (only for non-clinical)"
        )

    # exvivo_platform only for exvivo/exvivo-sim
    platform = evidence.get("platform") or {}
    if platform.get("exvivo_platform") and level not in ("exvivo", "exvivo-sim"):
        result.hard_errors.append(
            f"exvivo_platform present but biological_level={level} (only for exvivo/exvivo-sim)"
        )

    # Rule 11: clinical levels require disease_id
    if level in CLINICAL_LEVELS and not evidence.get("disease_id"):
        result.hard_errors.append(f"disease_id required for biological_level={level}")


def _rule_simulation_provenance(evidence: dict, result: ValidationResult) -> None:
    """Rule 12 (U2): sim_provenance validation."""
    level = evidence.get("biological_level", "")
    has_sim = evidence.get("simulation_provenance") is not None

    if has_sim and level not in SIM_LEVELS:
        result.hard_errors.append(
            f"simulation_provenance present but biological_level={level} (only for -sim levels)"
        )

    if level in SIM_LEVELS:
        sp = evidence.get("simulation_provenance") or {}
        if not sp.get("sim_model"):
            result.hard_errors.append(f"sim_model required for -sim level {level}")
        if not sp.get("sim_version"):
            result.hard_errors.append(f"sim_version required for -sim level {level}")


def _rule_readout(evidence: dict, result: ValidationResult) -> None:
    """Rule 13 (I6): readout_value required conditions."""
    has_readout_value = evidence.get("readout_value") is not None
    has_readout_unit = evidence.get("readout_unit") is not None
    is_two_arm = evidence.get("evidence_design") == "two_arm_comparison"
    arm_complete = _is_arm_stats_complete(evidence.get("arm_stats") or {})
    is_qual = evidence.get("is_qualitative") and evidence.get("readout_direction")

    # readout_value required unless two_arm with complete arm_stats, or qualitative
    if not has_readout_value and not is_qual:
        if not (is_two_arm and arm_complete):
            result.hard_errors.append(
                "readout_value missing, no complete arm_stats or qualitative triple (I6)"
            )

    if has_readout_value and not has_readout_unit:
        result.hard_errors.append("readout_unit required when readout_value present (I6)")


def _is_arm_stats_complete(arm: dict) -> bool:
    """Check if arm_stats is complete enough to omit readout_value."""
    kind = arm.get("kind", "")
    for role in ("intervention", "control"):
        data = arm.get(role) or {}
        if not data.get("n"):
            return False
        if kind == "continuous" and data.get("mean") is None:
            return False
    return True


def _rule_curies(evidence: dict, result: ValidationResult) -> None:
    """Rule 2: CURIE validation (recursive)."""

    def _validate_curie(curie_str: str, path: str) -> None:
        if ":" not in curie_str:
            result.hard_errors.append(f"CURIE '{curie_str}' at {path}: no prefix separator")
            return
        prefix, _, accession = curie_str.partition(":")
        if prefix not in ALL_PREFIXES:
            result.hard_errors.append(
                f"CURIE prefix '{prefix}' at {path} not in registered namespaces"
            )
            return
        if prefix in CURIE_PATTERNS and not CURIE_PATTERNS[prefix].match(accession):
            result.hard_errors.append(
                f"CURIE accession '{accession}' at {path} fails regex for prefix {prefix}"
            )

    def _scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # evidence_id is an internal content-addressed hash, not a CURIE
                if k == "evidence_id":
                    continue
                if k.endswith("_id") and isinstance(v, str) and v:
                    _validate_curie(v, f"{path}.{k}" if path else k)
                elif isinstance(v, (dict, list)):
                    _scan(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and v:
                    _scan(v, f"{path}[{i}]")

    _scan(evidence)


# ── evidence_id ────────────────────────────────────────────────────────────────


def _build_identity(evidence: dict) -> dict:
    """Build the identity object for evidence_id computation (I5)."""
    identity: dict = {}
    identity["biological_level"] = evidence.get("biological_level")

    # interventions: normalize to identity subset, sort by canonical_json
    interventions = evidence.get("interventions", [])
    identity_items = []
    for item in interventions:
        identity_items.append(
            {
                k: item.get(k)
                for k in ("role", "entity_type", "entity_id", "entity_label", "alteration")
            }
        )
    identity_items.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    identity["interventions"] = identity_items

    identity["disease_id"] = evidence.get("disease_id")

    # experimental_context subset
    ec = evidence.get("experimental_context") or {}
    ec_subset: dict = {}
    for k in ("model_method", "cell_line_id", "model_organism", "strain", "sex"):
        if k in ec and ec[k] is not None:
            ec_subset[k] = ec[k]
    # gene_background sorted
    gb = ec.get("gene_background") or []
    if gb:
        ec_subset["gene_background"] = sorted(
            [{"gene_id": g.get("gene_id"), "alteration": g.get("alteration")} for g in gb],
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )
    if ec_subset:
        identity["experimental_context"] = ec_subset

    # clinical_design subset
    cd = evidence.get("clinical_design") or {}
    cd_subset: dict = {}
    for k in ("comparator_type", "phase", "population", "study_id"):
        if k in cd and cd[k] is not None:
            cd_subset[k] = cd[k]
    if cd_subset:
        identity["clinical_design"] = cd_subset

    identity["readout_type"] = evidence.get("readout_type")

    sp = evidence.get("series_point")
    if sp:
        identity["series_point"] = sp

    # rank=1 source id
    for s in evidence.get("sources", []):
        if s.get("rank") == 1:
            identity["source_rank1"] = s.get("id", "")
            break

    return identity
