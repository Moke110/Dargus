"""D-Base v0.15.5 three-axis validator.

Implements spec §6 validation rules on the x/y/bg record shape.
Hard failures reject the write; soft failures set needs_curation=true.

Usage:
    from dargus.dbase.validate import validate_evidence, compute_evidence_id
    result = validate_evidence(evidence_dict)
    if not result.ok:
        raise ValidationError(result.hard_errors)
    if result.soft_warnings:
        evidence["needs_curation"] = True
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
    """Return frozenset of log-typed effect types."""
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
    return frozenset({"rct", "epi", "rct-sim"})


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
    """Run all §6 validation rules on a three-axis evidence record."""
    result = ValidationResult()

    _rule_string_nulls(evidence, result)
    _rule_biological_level(evidence, result)
    _rule_sources(evidence, result)
    _rule_xy_shape(evidence, result)
    _rule_x_axis(evidence, result)
    _rule_y_axis(evidence, result)
    _rule_bg(evidence, result)
    _rule_level_field_groups(evidence, result)
    _rule_simulation_provenance(evidence, result)
    _rule_curies(evidence, result)

    return result


def compute_evidence_id(evidence: dict) -> str:
    """Compute content-addressed evidence_id (§5)."""
    identity = _build_identity(evidence)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── rule: string nulls (§6.5 R-stringnull) ──────────────────────────────────


def _rule_string_nulls(evidence: dict, result: ValidationResult) -> None:
    string_nulls = frozenset({"null", "NA", "None", "nan", "N/A"})

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


# ── rule: biological_level (§6.1 R-level, R-design) ────────────────────────


def _rule_biological_level(evidence: dict, result: ValidationResult) -> None:
    level = evidence.get("biological_level")
    bls = _biological_levels()
    if level not in bls:
        result.hard_errors.append(f"biological_level '{level}' not in {sorted(bls)}")
        return

    # derive is_clinical (override any user value)
    evidence["is_clinical"] = 1 if level in _clinical_levels() else 0

    # R-design: evidence_design must be in vocab
    ed = evidence.get("evidence_design")
    designs = _vset("evidence_design")
    if ed and ed not in designs:
        result.hard_errors.append(f"evidence_design '{ed}' not in {sorted(designs)}")


# ── rule: sources (§6.5 R-sources) ──────────────────────────────────────────


def _rule_sources(evidence: dict, result: ValidationResult) -> None:
    sources = evidence.get("sources", [])
    if not sources:
        result.hard_errors.append("sources is empty")
        return

    rank1_count = sum(1 for s in sources if s.get("rank") == 1)
    if rank1_count != 1:
        result.hard_errors.append(f"sources must have exactly one rank=1, got {rank1_count}")

    source_types = _vset("source_type")
    for i, s in enumerate(sources):
        stype = s.get("type", "")
        if stype not in source_types:
            result.hard_errors.append(f"sources[{i}].type '{stype}' not in {sorted(source_types)}")
        sid = s.get("id", "")
        if not sid:
            result.hard_errors.append(f"sources[{i}].id is empty")
        else:
            _soft_validate_source_id(stype, sid, i, result)


def _soft_validate_source_id(stype: str, sid: str, idx: int, result: ValidationResult) -> None:
    patterns = {
        "doi": re.compile(r"^10\.\S+$"),
        "pmid": re.compile(r"^\d+$"),
        "pmcid": re.compile(r"^PMC\d+$"),
        "db_accession": re.compile(r"^[a-z_]+:.+$"),
    }
    if stype in patterns and not patterns[stype].match(sid):
        result.soft_warnings.append(f"sources[{idx}].id '{sid}' format mismatch for type {stype}")


# ── rule: xy shape (§6.1 structural + §6.1 R-count-design) ──────────────────


def _rule_xy_shape(evidence: dict, result: ValidationResult) -> None:
    xy = evidence.get("xy") or {}
    count = xy.get("count", 0)
    if not isinstance(count, int) or count < 0:
        result.hard_errors.append(f"xy.count must be int >= 0, got {count}")
        return

    # R-xcount: len(x.value) == xy.count
    xv = evidence.get("x", {}).get("value") or []
    if len(xv) != count and not (count == 0 and xv == []):
        result.hard_errors.append(f"len(x.value)={len(xv)} != xy.count={count} (R-xcount)")

    # R-ycount: len(y.value) == (1 if count==0 else count)
    yv = evidence.get("y", {}).get("value") or []
    expected_y_len = 1 if count == 0 else count
    if len(yv) != expected_y_len:
        result.hard_errors.append(f"len(y.value)={len(yv)} != {expected_y_len} (R-ycount)")

    # R-yarrays: parallel arrays must be [] or len==count
    y = evidence.get("y") or {}
    for arr_key in ("ci95", "dispersion", "n_total", "events", "p_value"):
        arr = y.get(arr_key)
        if arr is not None and len(arr) not in (0, count):
            result.hard_errors.append(
                f"y.{arr_key} length {len(arr)} != xy.count={count} (R-yarrays)"
            )

    # R-yvalue-num: every y.value element is a real number
    for i, v in enumerate(yv):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            result.hard_errors.append(f"y.value[{i}] = {v!r} is not a real number (R-yvalue-num)")

    # R-count-design: count ↔ design consistency
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


# ── rule: x axis (§6.2) ─────────────────────────────────────────────────────


def _rule_x_axis(evidence: dict, result: ValidationResult) -> None:
    x = evidence.get("x") or {}
    xtype = x.get("type", "")
    xvals = x.get("value") or []
    x_types = _vset("x_type")

    # R-xtype
    if xtype not in x_types:
        result.hard_errors.append(f"x.type '{xtype}' not in {sorted(x_types)}")
        return

    # R-xunit: x.unit only for numeric x-types
    xunit = x.get("unit")
    if xunit and xtype not in ("concentration", "time"):
        result.hard_errors.append(
            f"x.unit '{xunit}' set but x.type={xtype} (only for concentration/time)"
        )

    # Validate each x.value item
    control_labels = _vset("x_value_control_labels")
    alterations = _vset("alteration")
    design = evidence.get("evidence_design", "")

    for i, item in enumerate(xvals or []):
        if xtype in ("drug", "gene", "combination"):
            eid = item.get("entity_id")
            elabel = item.get("entity_label")
            if not eid and not elabel:
                # combination: at least one component needs entity_id/entity_label
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

            # R-alteration: only for gene x-type or bg.genes
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

    # R-xpairwise: two_arm_comparison must have x.value[0] != x.value[1]
    if design == "two_arm_comparison" and len(xvals) >= 2:
        if xvals[0] == xvals[1]:
            result.hard_errors.append(
                "x.value[0] == x.value[1] but evidence_design=two_arm_comparison (R-xpairwise)"
            )

    # R-xcontrol: for two_arm_comparison, x.value[1] must be control
    if design == "two_arm_comparison" and len(xvals) >= 2:
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

    # R-xmonotonic (soft): concentration/time SHOULD be strictly monotonic
    if xtype in ("concentration", "time") and len(xvals) >= 2:
        nums = [item.get("x") for item in xvals if isinstance(item.get("x"), (int, float))]
        if len(nums) == len(xvals):
            increasing = all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
            decreasing = all(nums[i] > nums[i + 1] for i in range(len(nums) - 1))
            if not (increasing or decreasing):
                result.soft_warnings.append(
                    "x.value[*].x is not strictly monotonic (R-xmonotonic soft)"
                )


# ── rule: y axis (§6.3) ─────────────────────────────────────────────────────


def _rule_y_axis(evidence: dict, result: ValidationResult) -> None:
    y = evidence.get("y") or {}
    ycat = y.get("category", "")
    ytype = y.get("type", "")
    design = evidence.get("evidence_design", "")

    # R-ycat
    y_categories = _vset("y_category")
    if ycat not in y_categories:
        result.hard_errors.append(f"y.category '{ycat}' not in {sorted(y_categories)}")
    if not ytype or not isinstance(ytype, str):
        result.hard_errors.append("y.type must be non-empty string")

    # R-ydir: required for comparative designs
    ydir = y.get("direction")
    comparative = ("two_arm_comparison", "observational_association")
    if design in comparative and not ydir:
        result.hard_errors.append(f"y.direction required for evidence_design={design} (R-ydir)")
    if ydir and ydir not in _vset("y_direction"):
        result.hard_errors.append(f"y.direction '{ydir}' not in {sorted(_vset('y_direction'))}")

    # R-effect
    effect = y.get("effect") or {}
    if effect:
        if "value" not in effect or "type" not in effect:
            result.hard_errors.append("y.effect must have 'value' and 'type'")
        else:
            etype = effect.get("type", "")
            if etype not in _vset("y_effect_type"):
                result.hard_errors.append(
                    f"y.effect.type '{etype}' not in {sorted(_vset('y_effect_type'))}"
                )
            escale = effect.get("scale", "linear")
            if escale not in _vset("y_effect_scale"):
                result.hard_errors.append(
                    f"y.effect.scale '{escale}' not in {sorted(_vset('y_effect_scale'))}"
                )
            log_types = _vlog("y_effect_type")
            if etype in log_types and escale != "log":
                result.hard_errors.append(
                    f"y.effect.type '{etype}' requires scale='log', got '{escale}'"
                )
            eff_ci = effect.get("ci95") or {}
            if eff_ci:
                low = eff_ci.get("lower")
                up = eff_ci.get("upper")
                if low is not None and up is not None and low > up:
                    result.hard_errors.append(f"y.effect.ci95 lower {low} > upper {up}")

    # R-ci: per-point CI validation
    yvals = y.get("value") or []
    for i, ci in enumerate(y.get("ci95") or []):
        if not ci:
            continue
        low = ci.get("lower")
        up = ci.get("upper")
        if low is not None and up is not None and low > up:
            result.hard_errors.append(f"y.ci95[{i}] lower {low} > upper {up}")
        # soft: point should be within CI
        if i < len(yvals) and low is not None and up is not None:
            pt = yvals[i]
            if not (low <= pt <= up):
                result.soft_warnings.append(f"y.ci95[{i}] point {pt} not in [{low}, {up}]")

    # R-pval: each in [0,1]
    for i, pv in enumerate(y.get("p_value") or []):
        if pv is not None and not (0 <= pv <= 1):
            result.hard_errors.append(f"y.p_value[{i}]={pv} not in [0,1]")

    # R-events: 0 <= events[i] <= n_total[i]
    events = y.get("events") or []
    n_totals = y.get("n_total") or []
    for i, ev in enumerate(events):
        if ev is not None:
            n = n_totals[i] if i < len(n_totals) else None
            if n is not None and not (0 <= ev <= n):
                result.hard_errors.append(f"y.events[{i}]={ev} not in [0, n_total={n}]")

    # R-ybasis: if present, must be valid
    ybasis = y.get("basis")
    if ybasis and ybasis not in _vset("y_basis"):
        result.hard_errors.append(f"y.basis '{ybasis}' not in {sorted(_vset('y_basis'))}")


# ── rule: bg axis ────────────────────────────────────────────────────────────


def _rule_bg(evidence: dict, result: ValidationResult) -> None:
    bg = evidence.get("bg") or {}
    level = evidence.get("biological_level", "")
    clinical = _clinical_levels()

    # clinical levels MUST have non-empty bg.disease_id
    if level in clinical:
        dids = bg.get("disease_id") or []
        if not dids:
            result.hard_errors.append(f"bg.disease_id required for biological_level={level} (§6.4)")

    # validate bg.genes[*].alteration
    alterations = _vset("alteration")
    for i, gene in enumerate(bg.get("genes") or []):
        alt = gene.get("alteration")
        if alt is not None and alt not in alterations:
            result.hard_errors.append(
                f"bg.genes[{i}].alteration '{alt}' not in {sorted(alterations)}"
            )


# ── rule: level ↔ field groups (§6.4) ───────────────────────────────────────


def _rule_level_field_groups(evidence: dict, result: ValidationResult) -> None:
    level = evidence.get("biological_level", "")
    clinical = _clinical_levels()
    non_clinical = _biological_levels() - clinical

    # clinical_design only for clinical levels
    if evidence.get("clinical_design") and level not in clinical:
        result.hard_errors.append(
            f"clinical_design present but biological_level={level} (only for clinical)"
        )

    # bg.model / bg.genes / cell_line_id / cell_type / assay_platform / exvivo_platform
    # disallowed for clinical levels
    if level in clinical:
        bg = evidence.get("bg") or {}
        if bg.get("model"):
            result.hard_errors.append(
                f"bg.model present but biological_level={level} (only for non-clinical)"
            )
        if bg.get("genes"):
            result.hard_errors.append(
                f"bg.genes present but biological_level={level} (only for non-clinical)"
            )
        if evidence.get("cell_line_id"):
            result.hard_errors.append(
                f"cell_line_id present but biological_level={level} (only for non-clinical)"
            )
        if evidence.get("cell_type"):
            result.hard_errors.append(
                f"cell_type present but biological_level={level} (only for non-clinical)"
            )
        if evidence.get("assay_platform"):
            result.hard_errors.append(
                f"assay_platform present but biological_level={level} (only for non-clinical)"
            )
        if evidence.get("exvivo_platform"):
            result.hard_errors.append(
                f"exvivo_platform present but biological_level={level} (only for non-clinical)"
            )

    # non-clinical levels MUST NOT carry clinical_design
    if level in non_clinical and evidence.get("clinical_design"):
        result.hard_errors.append(
            f"clinical_design present but biological_level={level} (only for clinical)"
        )

    # exvivo_platform only for exvivo/exvivo-sim
    exvivo_levels = {"exvivo", "exvivo-sim"}
    if evidence.get("exvivo_platform") and level not in exvivo_levels:
        result.hard_errors.append(
            f"exvivo_platform present but biological_level={level} (only for exvivo/exvivo-sim)"
        )


# ── rule: simulation provenance (§6.4) ──────────────────────────────────────


def _rule_simulation_provenance(evidence: dict, result: ValidationResult) -> None:
    level = evidence.get("biological_level", "")
    sim_levels = _sim_levels()
    sp = evidence.get("simulation_provenance")

    if sp and level not in sim_levels:
        result.hard_errors.append(
            f"simulation_provenance present but biological_level={level} (only for -sim levels)"
        )

    if level in sim_levels:
        if not sp:
            result.hard_errors.append(f"simulation_provenance required for -sim level {level}")
        else:
            if not sp.get("sim_model"):
                result.hard_errors.append("simulation_provenance.sim_model required for -sim level")
            if not sp.get("version"):
                result.hard_errors.append("simulation_provenance.version required for -sim level")


# ── rule: CURIE validation (§6.5 R-curie) ───────────────────────────────────


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


# ── evidence_id identity (§5) ────────────────────────────────────────────────


def _build_identity(evidence: dict) -> dict:
    """Build the identity object for evidence_id computation (§5).

    Identity keys:
      biological_level, x.type, x.value (normalized), y.type,
      bg.disease_id (sorted), clinical_design subset,
      cell_line_id, model_organism, strain, sex, exposure (dose_value + unit),
      experiment_group_id, source_rank1 id.
    """
    identity: dict = {}
    identity["biological_level"] = evidence.get("biological_level")

    # x axis identity
    x = evidence.get("x") or {}
    identity["x_type"] = x.get("type")

    # x.value normalized for identity
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

        # gene alteration
        alt = item.get("alteration")
        if xtype == "gene" and alt:
            ident_item["alteration"] = alt

        # dose
        dose = item.get("dose")
        if dose and isinstance(dose, dict):
            ident_item["dose"] = {"v": dose.get("v"), "u": dose.get("u")}

        # combination components
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

    # bg.disease_id (sorted)
    bg = evidence.get("bg") or {}
    dids = bg.get("disease_id") or []
    if dids:
        identity["bg.disease_id"] = sorted(dids)

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

    # exposure
    if evidence.get("exposure_dose_value") is not None:
        identity["exposure"] = [
            evidence["exposure_dose_value"],
            evidence.get("exposure_dose_unit"),
        ]

    # experiment_group_id
    egid = evidence.get("experiment_group_id")
    if egid:
        identity["experiment_group_id"] = egid

    # source_rank1
    for s in evidence.get("sources", []):
        if s.get("rank") == 1:
            identity["source_rank1"] = s.get("id", "")
            break

    return identity
