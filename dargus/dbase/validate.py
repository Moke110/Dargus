"""D-Base v1.0.0 three-axis validator.

Implements the design/2.1.x validation rules on the x/y/bg record shape.
Hard failures reject the write; soft failures set needs_curation=true.

Usage:
    from dargus.dbase.validate import validate_evidence, compute_evidence_id
    result = validate_evidence(evidence_dict)
    if not result.ok:
        raise ValidationError(result.hard_errors)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── vocabulary (loaded from vocabularies.json) ───────────────────────────────

_vocab: dict = {}


def _load_vocab() -> dict:
    global _vocab
    if _vocab:
        return _vocab
    vocab_path = Path(__file__).resolve().parent / "vocabularies.json"
    if vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as fh:
            _vocab = json.load(fh)
    return _vocab


def _v(key: str) -> list | dict:
    v = _load_vocab()
    return v.get(key, {})


def _vset(name: str) -> frozenset:
    """Return frozenset of enum values from vocabularies.json (list format)."""
    entry = _v(name)
    if isinstance(entry, dict) and "values" in entry:
        vals = entry["values"]
        if isinstance(vals, list) and vals and isinstance(vals[0], dict):
            return frozenset(item["value"] for item in vals)
        return frozenset(vals)
    return frozenset()


def _vlog(name: str) -> frozenset:
    """Return frozenset of log-typed effect value types."""
    entry = _v(name)
    if isinstance(entry, dict) and "log_types" in entry:
        return frozenset(entry["log_types"])
    return frozenset()


# ── vocabulary sets ──────────────────────────────────────────────────────────


def _biological_levels() -> frozenset:
    return _vset("biological_level")


def _clinical_levels() -> frozenset:
    v = _v("biological_level")
    if isinstance(v, dict) and "values" in v:
        return frozenset(item["value"] for item in v["values"] if item.get("is_clinical"))
    return frozenset({"rct", "epi"})


def _sim_levels() -> frozenset:
    v = _v("biological_level")
    if isinstance(v, dict) and "values" in v:
        return frozenset(item["value"] for item in v["values"] if item.get("is_sim"))
    return frozenset()


def _curie_hard() -> dict:
    v = _v("curie_prefixes")
    if isinstance(v, dict) and "hard_validated" in v:
        return v["hard_validated"]
    return {}


def _curie_fallback() -> frozenset:
    v = _v("curie_prefixes")
    if isinstance(v, dict) and "fallback" in v:
        return frozenset(v["fallback"])
    return frozenset()


# ── compile CURIE patterns once ──────────────────────────────────────────────

_CURIE_PATTERNS: dict[str, re.Pattern] = {}


def _curie_patterns() -> dict[str, re.Pattern]:
    global _CURIE_PATTERNS
    if _CURIE_PATTERNS:
        return _CURIE_PATTERNS
    for prefix, pattern_str in _curie_hard().items():
        _CURIE_PATTERNS[prefix] = re.compile(pattern_str)
    return _CURIE_PATTERNS


# ── ValidationResult ─────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    hard_errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.hard_errors) == 0


# ── public API ───────────────────────────────────────────────────────────────


def validate_evidence(evidence: dict) -> ValidationResult:
    """Run all validation rules on a three-axis evidence record."""
    result = ValidationResult()

    _rule_string_nulls(evidence, result)
    _rule_biological_level(evidence, result)
    _rule_sources(evidence, result)
    _rule_xy_shape(evidence, result)
    _rule_x_axis(evidence, result)
    _rule_y_axis(evidence, result)
    _rule_bg(evidence, result)
    _rule_level_field_groups(evidence, result)
    _rule_curies(evidence, result)

    return result


def compute_evidence_id(evidence: dict) -> str:
    """Compute content-addressed evidence_id from the identity fields."""
    identity = _build_identity(evidence)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── rule: string nulls ───────────────────────────────────────────────────────


def _rule_string_nulls(evidence: dict, result: ValidationResult) -> None:
    string_nulls = frozenset(_vset("string_nulls") or {"null", "NA", "None", "nan", "N/A"})

    def _scan(obj, path="root"):
        if isinstance(obj, str):
            if obj in string_nulls:
                result.hard_errors.append(f"String null '{obj}' at {path}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]")

    _scan(evidence)


# ── rule: biological_level ───────────────────────────────────────────────────


def _rule_biological_level(evidence: dict, result: ValidationResult) -> None:
    level = evidence.get("biological_level")
    bls = _biological_levels()
    if level not in bls:
        result.hard_errors.append(f"biological_level '{level}' not in {sorted(bls)}")
        return

    # derive is_clinical (override any user value); rct-sim is non-clinical
    evidence["is_clinical"] = 1 if level in _clinical_levels() else 0

    ed = evidence.get("evidence_design")
    designs = _vset("evidence_design")
    if ed and ed not in designs:
        result.hard_errors.append(f"evidence_design '{ed}' not in {sorted(designs)}")


# ── rule: sources / source_entry / source_time ───────────────────────────────


def _rule_sources(evidence: dict, result: ValidationResult) -> None:
    sources = evidence.get("sources", [])
    if not sources:
        result.hard_errors.append("sources is empty")
    else:
        rank1_count = sum(1 for s in sources if s.get("rank") == 1)
        if rank1_count != 1:
            result.hard_errors.append(f"sources must have exactly one rank=1, got {rank1_count}")

        source_types = _vset("source_type")
        for i, s in enumerate(sources):
            stype = s.get("type", "")
            if stype not in source_types:
                result.hard_errors.append(
                    f"sources[{i}].type '{stype}' not in {sorted(source_types)}"
                )
            if not s.get("name"):
                result.hard_errors.append(f"sources[{i}].name is empty")

    if not evidence.get("source_entry"):
        result.hard_errors.append("source_entry is required")
    if not evidence.get("source_time"):
        result.hard_errors.append("source_time is required")


# ── rule: xy shape ───────────────────────────────────────────────────────────


def _rule_xy_shape(evidence: dict, result: ValidationResult) -> None:
    xy = evidence.get("xy") or {}
    count = xy.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        result.hard_errors.append(f"xy.count must be int >= 1, got {count}")
        return

    xv = evidence.get("x", {}).get("value") or []
    if len(xv) != count:
        result.hard_errors.append(f"len(x.value)={len(xv)} != xy.count={count} (R-xcount)")

    y = evidence.get("y") or {}
    yv = y.get("value") or []
    if len(yv) != count:
        result.hard_errors.append(f"len(y.value)={len(yv)} != xy.count={count} (R-ycount)")

    for arr_key in ("dispersion", "n_total", "events", "p_value"):
        arr = y.get(arr_key)
        if arr is not None and len(arr) not in (0, count):
            result.hard_errors.append(
                f"y.{arr_key} length {len(arr)} != xy.count={count} (R-yarrays)"
            )

    for i, v in enumerate(yv):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            result.hard_errors.append(f"y.value[{i}] = {v!r} is not a real number (R-yvalue-num)")

    design = evidence.get("evidence_design", "")
    count_rules = _v("count_design_rules") if isinstance(_v("count_design_rules"), dict) else {}
    rules = count_rules.get(design, {})
    cmin = rules.get("min")
    cmax = rules.get("max")
    if cmin is not None and count < cmin:
        result.hard_errors.append(
            f"evidence_design={design} requires xy.count >= {cmin}, got {count}"
        )
    if cmax is not None and count > cmax:
        result.hard_errors.append(
            f"evidence_design={design} requires xy.count <= {cmax}, got {count}"
        )


# ── rule: x axis ─────────────────────────────────────────────────────────────


def _rule_x_axis(evidence: dict, result: ValidationResult) -> None:
    x = evidence.get("x") or {}
    xtype = x.get("type", "")
    xvals = x.get("value") or []
    x_types = _vset("x_type")

    if xtype not in x_types:
        result.hard_errors.append(f"x.type '{xtype}' not in {sorted(x_types)}")
        return

    xunit = x.get("unit")
    if xtype in ("concentration", "time") and not xunit:
        result.hard_errors.append(f"x.unit required for x.type={xtype}")
    if xunit and xtype not in ("concentration", "time"):
        result.hard_errors.append(
            f"x.unit '{xunit}' set but x.type={xtype} (only for concentration/time)"
        )

    control_labels = _vset("x_value_control_labels")
    alterations = _vset("alteration")
    design = evidence.get("evidence_design", "")

    for i, item in enumerate(xvals or []):
        if xtype in ("drug", "gene", "combination"):
            eid = item.get("entity_id")
            elabel = item.get("entity_label")
            if not eid and not elabel:
                components = item.get("components") or []
                if xtype == "combination":
                    has_id = any(c.get("entity_id") or c.get("entity_label") for c in components)
                    if not has_id:
                        result.hard_errors.append(
                            f"x.value[{i}]: no entity_id/entity_label (R-xtype-entity)"
                        )
                else:
                    result.hard_errors.append(
                        f"x.value[{i}]: entity_id and entity_label both empty (R-xtype-entity)"
                    )

            alt = item.get("alteration")
            if alt is not None:
                if xtype != "gene":
                    result.hard_errors.append(
                        f"x.value[{i}].alteration on non-gene x.type='{xtype}'"
                    )
                elif alt not in alterations:
                    result.hard_errors.append(
                        f"x.value[{i}].alteration '{alt}' not in {sorted(alterations)}"
                    )

        elif xtype in ("concentration", "time"):
            val = item.get("x")
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                result.hard_errors.append(
                    f"x.value[{i}].x must be numeric for x.type={xtype} (R-xtype-num)"
                )

    if design == "two_arm_comparison" and len(xvals) >= 2:
        if xvals[0] == xvals[1]:
            result.hard_errors.append(
                "x.value[0] == x.value[1] but evidence_design=two_arm_comparison (R-xpairwise)"
            )
        ctrl = xvals[1]
        if ctrl.get("entity_id") is not None:
            result.hard_errors.append(
                f"x.value[1] (control) must have entity_id=null, got {ctrl.get('entity_id')}"
            )
        if ctrl.get("entity_label", "") not in control_labels:
            result.soft_warnings.append(
                f"x.value[1] control label '{ctrl.get('entity_label')}' "
                f"not in sanctioned set {sorted(control_labels)} (R-xcontrol soft)"
            )

    if xtype in ("concentration", "time") and len(xvals) >= 2:
        nums = [item.get("x") for item in xvals if isinstance(item.get("x"), (int, float))]
        if len(nums) == len(xvals):
            increasing = all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
            decreasing = all(nums[i] > nums[i + 1] for i in range(len(nums) - 1))
            if not (increasing or decreasing):
                result.soft_warnings.append(
                    "x.value[*].x is not strictly monotonic (R-xmonotonic soft)"
                )


# ── rule: y axis ─────────────────────────────────────────────────────────────


def _rule_y_axis(evidence: dict, result: ValidationResult) -> None:
    y = evidence.get("y") or {}
    ycat = y.get("category", "")
    ytype = y.get("type", "")
    design = evidence.get("evidence_design", "")

    y_categories = _vset("y_category")
    if ycat not in y_categories:
        result.hard_errors.append(f"y.category '{ycat}' not in {sorted(y_categories)}")
    if not ytype or not isinstance(ytype, str):
        result.hard_errors.append("y.type must be non-empty string")

    ydir = y.get("direction")
    comparative = ("two_arm_comparison", "observational_association")
    if design in comparative and not ydir:
        result.hard_errors.append(f"y.direction required for evidence_design={design} (R-ydir)")
    if ydir and ydir not in _vset("y_direction"):
        result.hard_errors.append(f"y.direction '{ydir}' not in {sorted(_vset('y_direction'))}")

    # y.effect: {value, value_type, dispersion, dispersion_type}
    effect = y.get("effect") or {}
    if effect:
        if "value" not in effect or "value_type" not in effect:
            result.hard_errors.append("y.effect must have 'value' and 'value_type'")
        else:
            vtype = effect.get("value_type", "")
            if vtype not in _vset("y_effect_value_type"):
                result.hard_errors.append(
                    f"y.effect.value_type '{vtype}' not in {sorted(_vset('y_effect_value_type'))}"
                )
            dtype = effect.get("dispersion_type")
            if dtype is not None and dtype not in _vset("y_effect_dispersion_type"):
                result.hard_errors.append(
                    f"y.effect.dispersion_type '{dtype}' "
                    f"not in {sorted(_vset('y_effect_dispersion_type'))}"
                )
            if vtype in _vlog("y_effect_value_type"):
                # log measures: value on log scale; only a soft sanity note possible
                pass

    # y.dispersion entries: {type, value}
    for i, disp in enumerate(y.get("dispersion") or []):
        if not isinstance(disp, dict):
            result.hard_errors.append(f"y.dispersion[{i}] must be an object")
            continue
        dt = disp.get("type")
        if dt not in _vset("y_dispersion_type"):
            result.hard_errors.append(
                f"y.dispersion[{i}].type '{dt}' not in {sorted(_vset('y_dispersion_type'))}"
            )
        if "value" not in disp:
            result.hard_errors.append(f"y.dispersion[{i}].value missing")

    for i, pv in enumerate(y.get("p_value") or []):
        if pv is not None and not (0 <= pv <= 1):
            result.hard_errors.append(f"y.p_value[{i}]={pv} not in [0,1]")

    events = y.get("events") or []
    n_totals = y.get("n_total") or []
    for i, ev in enumerate(events):
        if ev is not None:
            n = n_totals[i] if i < len(n_totals) else None
            if n is not None and not (0 <= ev <= n):
                result.hard_errors.append(f"y.events[{i}]={ev} not in [0, n_total={n}]")

    to_basis = y.get("to_basis")
    if to_basis and to_basis not in _vset("y_to_basis"):
        result.hard_errors.append(f"y.to_basis '{to_basis}' not in {sorted(_vset('y_to_basis'))}")


# ── rule: bg axis ────────────────────────────────────────────────────────────


def _rule_bg(evidence: dict, result: ValidationResult) -> None:
    bg = evidence.get("bg") or {}
    level = evidence.get("biological_level", "")
    clinical = _clinical_levels()

    if level in clinical:
        dids = bg.get("disease_id") or []
        if not dids:
            result.hard_errors.append(f"bg.disease_id required for biological_level={level}")

    alterations = _vset("alteration")
    for i, gene in enumerate(bg.get("genes") or []):
        alt = gene.get("alteration")
        if alt is not None and alt not in alterations:
            result.hard_errors.append(
                f"bg.genes[{i}].alteration '{alt}' not in {sorted(alterations)}"
            )

    dose_value = bg.get("dose_value")
    dose_unit = bg.get("dose_unit")
    if (dose_value is None) != (dose_unit is None):
        result.soft_warnings.append("bg.dose_value and bg.dose_unit should be set together")
    duration_value = bg.get("duration_value")
    duration_unit = bg.get("duration_unit")
    if (duration_value is None) != (duration_unit is None):
        result.soft_warnings.append("bg.duration_value and bg.duration_unit should be set together")


# ── rule: level ↔ field groups ───────────────────────────────────────────────


def _rule_level_field_groups(evidence: dict, result: ValidationResult) -> None:
    level = evidence.get("biological_level", "")
    clinical = _clinical_levels()
    y = evidence.get("y") or {}

    if evidence.get("clinical_design") and level not in clinical:
        result.hard_errors.append(
            f"clinical_design present but biological_level={level} (only for clinical)"
        )

    if level in clinical:
        bg = evidence.get("bg") or {}
        if bg.get("model"):
            result.hard_errors.append(
                f"bg.model present but biological_level={level} (only for non-clinical)"
            )
        if evidence.get("cell_line_id"):
            result.hard_errors.append(
                f"cell_line_id present but biological_level={level} (only for non-clinical)"
            )

    if level and level not in clinical and not y.get("assay"):
        result.soft_warnings.append(f"y.assay recommended for non-clinical level {level}")

    exvivo_levels = {"exvivo", "exvivo-sim"}
    if level in exvivo_levels and not evidence.get("exvivo_platform"):
        result.hard_errors.append(f"exvivo_platform required for biological_level={level}")
    if evidence.get("exvivo_platform") and level not in exvivo_levels:
        result.hard_errors.append(
            f"exvivo_platform present but biological_level={level} (only for exvivo/exvivo-sim)"
        )


# ── rule: CURIE validation ───────────────────────────────────────────────────


def _rule_curies(evidence: dict, result: ValidationResult) -> None:
    patterns = _curie_patterns()
    fallback = _curie_fallback()
    all_prefixes = frozenset(patterns.keys()) | fallback

    def _validate_curie(curie_str: str, path: str) -> None:
        if ":" not in curie_str:
            result.hard_errors.append(f"CURIE '{curie_str}' at {path}: no prefix separator")
            return
        prefix, _, accession = curie_str.partition(":")
        if prefix not in all_prefixes:
            result.hard_errors.append(f"CURIE prefix '{prefix}' at {path} not registered")
            return
        if prefix in patterns and not patterns[prefix].match(accession):
            result.hard_errors.append(
                f"CURIE accession '{accession}' at {path} fails regex for {prefix}"
            )

    def _scan(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "evidence_id":
                    continue
                if k.endswith("_id") and isinstance(v, str) and v:
                    _validate_curie(v, f"{path}.{k}")
                elif isinstance(v, (dict, list)):
                    _scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and v:
                    _validate_curie(v, f"{path}[{i}]")
                elif isinstance(v, dict):
                    for dk, dv in v.items():
                        if dk.endswith("_id") and isinstance(dv, str) and dv:
                            _validate_curie(dv, f"{path}[{i}].{dk}")

    _scan(evidence)


# ── evidence_id identity ─────────────────────────────────────────────────────


def _build_identity(evidence: dict) -> dict:
    """Build the identity object for evidence_id computation.

    Identity fields (design/2.1.1 in_id = Y):
      biological_level, sources (sorted), source_entry, source_time,
      x.type, x.value (normalized), y.type,
      bg.disease_id (sorted), bg.dose_value, bg.dose_unit,
      cell_line_id, model_organism, strain, sex,
      clinical_design.{comparator_type, phase, population, study_id},
      related_evidence_id (sorted).
    """
    identity: dict = {}
    identity["biological_level"] = evidence.get("biological_level")

    # provenance identity
    sources = evidence.get("sources") or []
    norm_sources = sorted(
        ({"rank": s.get("rank"), "type": s.get("type"), "name": s.get("name")} for s in sources),
        key=lambda s: json.dumps(s, sort_keys=True),
    )
    identity["sources"] = norm_sources
    identity["source_entry"] = evidence.get("source_entry")
    identity["source_time"] = evidence.get("source_time")

    # x axis identity
    x = evidence.get("x") or {}
    identity["x_type"] = x.get("type")

    xvals = x.get("value") or []
    identity_x = []
    control_labels = _vset("x_value_control_labels")
    xtype = x.get("type", "")
    for item in xvals:
        eid = item.get("entity_id", None)
        elabel = item.get("entity_label", None)

        # control arm normalization: entity_id=null + sanctioned label → __control__
        if eid is None and elabel in control_labels:
            identity_x.append({"__control__": True})
            continue

        ident_item: dict = {}
        if eid:
            ident_item["entity_id"] = eid
        elif elabel:
            ident_item["entity_label"] = elabel

        alt = item.get("alteration")
        if xtype == "gene" and alt:
            ident_item["alteration"] = alt

        dose = item.get("dose")
        if dose and isinstance(dose, dict):
            ident_item["dose"] = {"v": dose.get("v"), "u": dose.get("u")}

        components = item.get("components")
        if xtype == "combination" and components:
            comp_ident = []
            for c in components:
                ci: dict = {}
                if c.get("entity_id"):
                    ci["entity_id"] = c["entity_id"]
                elif c.get("entity_label"):
                    ci["entity_label"] = c["entity_label"]
                cd = c.get("dose")
                if cd and isinstance(cd, dict):
                    ci["dose"] = {"v": cd.get("v"), "u": cd.get("u")}
                comp_ident.append(ci)
            ident_item["components"] = comp_ident

        identity_x.append(ident_item)
    identity["x_value"] = identity_x

    # y axis identity (only y.type)
    identity["y_type"] = (evidence.get("y") or {}).get("type")

    # bg identity
    bg = evidence.get("bg") or {}
    dids = bg.get("disease_id") or []
    if dids:
        identity["bg.disease_id"] = sorted(dids)
    if bg.get("dose_value") is not None:
        identity["bg.dose"] = [bg["dose_value"], bg.get("dose_unit")]

    # clinical_design subset
    cd = evidence.get("clinical_design") or {}
    cd_subset: dict = {}
    for k in ("comparator_type", "phase", "population", "study_id"):
        if k in cd and cd[k] is not None:
            cd_subset[k] = cd[k]
    if cd_subset:
        identity["clinical_design"] = cd_subset

    # sample identity
    for k in ("cell_line_id", "model_organism", "strain", "sex"):
        if k in evidence and evidence[k] is not None:
            identity[k] = evidence[k]

    # cross-record links
    rel = evidence.get("related_evidence_id") or []
    if rel:
        identity["related_evidence_id"] = sorted(rel)

    return identity
