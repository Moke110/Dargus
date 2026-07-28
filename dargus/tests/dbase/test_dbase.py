"""Tests for DBase v1.0.0 — shard JSONL evidence store with three-axis records."""

import tempfile

from dargus.dbase import DBase
from dargus.dbase.validate import compute_evidence_id, validate_evidence


def _make_evidence(**overrides):
    """Return a valid v1.0.0 three-axis evidence dict (descriptive, xy.count=1)."""
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {
            "type": "logP",
            "category": "pk_adme",
            "value": [3.5],
        },
        "bg": {"disease_id": [], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


def test_dbase_append_and_read_shards():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        e = _make_evidence(evidence_id="ev_test123")
        dbase.append_shard(e)
        records = dbase.read_shards()
        assert len(records) == 1
        assert records[0]["evidence_id"] == "ev_test123"


def test_dbase_evidence_id_exists():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        e = _make_evidence(evidence_id="ev_real")
        dbase.append_shard(e)
        assert dbase.evidence_id_exists("ev_real")
        assert not dbase.evidence_id_exists("ev_nonexistent")


def test_dbase_clear():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        dbase.append_shard(_make_evidence(evidence_id="ev_1"))
        assert len(dbase.read_shards()) == 1
        dbase.clear()
        assert len(dbase.read_shards()) == 0


def test_validate_evidence_ok():
    result = validate_evidence(_make_evidence())
    assert result.ok, result.hard_errors


def test_validate_evidence_rejects_empty_sources():
    result = validate_evidence(_make_evidence(sources=[]))
    assert not result.ok


def test_validate_evidence_rejects_invalid_level():
    result = validate_evidence(_make_evidence(biological_level="clinical"))
    assert not result.ok


def test_validate_requires_source_entry_and_time():
    e = _make_evidence()
    del e["source_entry"]
    del e["source_time"]
    result = validate_evidence(e)
    assert not result.ok
    assert any("source_entry" in err for err in result.hard_errors)
    assert any("source_time" in err for err in result.hard_errors)


def test_evidence_id_stable():
    """Evidence_id is content-addressed and stable."""
    e = {
        "biological_level": "rct",
        "evidence_design": "two_arm_comparison",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {"entity_id": "chembl:CHEMBL25", "entity_label": "test_drug"},
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        "y": {
            "type": "updrs_iii_change",
            "category": "clinic_efficacy_primary",
            "unit": "points",
            "value": [-5.0, 0.5],
            "direction": "beneficial",
        },
        "bg": {"disease_id": ["mondo:0005180"], "drugs": [], "genes": []},
        "clinical_design": {
            "comparator_type": "placebo",
            "phase": "phase_3",
            "population": "adults",
            "study_id": "clinicaltrials:NCT01234567",
        },
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    assert compute_evidence_id(e) == compute_evidence_id(e)
    assert compute_evidence_id(e).startswith("ev_")


def test_evidence_id_distinct_for_different_y_type():
    e1 = _make_evidence()
    e2 = _make_evidence()
    e2["y"] = dict(e1["y"], type="solubility")
    assert compute_evidence_id(e1) != compute_evidence_id(e2)


def test_validate_xy_count_mismatch():
    """x.value length != xy.count should be rejected."""
    result = validate_evidence(
        _make_evidence(
            xy={"count": 2},
            x={"type": "drug", "value": []},
            y={"type": "test", "category": "other", "value": [1.0, 2.0]},
            evidence_design="two_arm_comparison",
        )
    )
    assert not result.ok
    assert any("R-xcount" in e for e in result.hard_errors)


def test_validate_y_category_invalid():
    result = validate_evidence(
        _make_evidence(y={"type": "test", "category": "not_a_real_category", "value": [1.0]})
    )
    assert not result.ok


def test_dbase_global_instance():
    dbase = DBase.global_instance()
    assert dbase.project_id == "global"


def test_validate_control_arm_must_have_null_entity_id():
    result = validate_evidence(
        {
            "biological_level": "rct",
            "evidence_design": "two_arm_comparison",
            "xy": {"count": 2},
            "x": {
                "type": "drug",
                "value": [
                    {"entity_id": "chembl:CHEMBL25", "entity_label": "drug"},
                    {"entity_id": "chembl:CHEMBL26", "entity_label": "not_control"},
                ],
            },
            "y": {
                "type": "test",
                "category": "clinic_efficacy_primary",
                "value": [1.0, 2.0],
                "direction": "beneficial",
            },
            "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
            "clinical_design": {
                "comparator_type": "placebo",
                "phase": "phase_3",
                "population": "adults",
                "study_id": "clinicaltrials:NCT01234567",
            },
            "sources": [{"rank": 1, "type": "journal", "name": "PMID 12345678"}],
            "source_entry": "PMID:12345678",
            "source_time": "2025-05-01",
        }
    )
    assert not result.ok
    assert any("control" in e.lower() for e in result.hard_errors)


def test_validate_descriptive_requires_count_1():
    """Descriptive records have xy.count=1 per design/2.1.2."""
    e = _make_evidence(xy={"count": 0})
    result = validate_evidence(e)
    assert not result.ok


def test_validate_effect_value_type_vocab():
    """y.effect uses value_type from the 16-measure design vocabulary."""
    e = _make_evidence()
    e["y"]["effect"] = {"value": 1.5, "value_type": "OR"}
    assert validate_evidence(e).ok
    e["y"]["effect"] = {"value": 1.5, "value_type": "IC50"}
    assert not validate_evidence(e).ok
    e["y"]["effect"] = {"value": 1.5, "type": "OR"}  # old schema key
    assert not validate_evidence(e).ok


def test_validate_y_to_basis_vocab():
    e = _make_evidence()
    e["y"]["to_basis"] = "log2_fold_change"
    assert validate_evidence(e).ok
    e["y"]["to_basis"] = "ratio_to_baseline"  # old vocab value
    assert not validate_evidence(e).ok


def test_validate_dispersion_entries():
    e = _make_evidence()
    e["y"]["dispersion"] = [{"type": "CI95", "value": [3.0, 4.0]}]
    assert validate_evidence(e).ok
    e["y"]["dispersion"] = [{"type": "BOGUS", "value": 1.0}]
    assert not validate_evidence(e).ok


# ── S7_T2: CURIE validation driven by field_registry.yaml ──────────────────────


def test_curie_related_evidence_id_passes_with_plain_ids():
    """related_evidence_id is type: str[], not type: curie — plain ev_* IDs must pass."""
    e = _make_evidence()
    e["related_evidence_id"] = ["ev_abc123", "ev_def456"]
    result = validate_evidence(e)
    assert result.ok, result.hard_errors


def test_curie_malformed_study_id_rejected():
    """clinical_design.study_id is type: curie — a value without ':' must fail."""
    e = _make_evidence(
        biological_level="rct",
        evidence_design="two_arm_comparison",
        xy={"count": 2},
        x={
            "type": "drug",
            "value": [
                {"entity_id": "chembl:CHEMBL25", "entity_label": "drug"},
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        y={
            "type": "test",
            "category": "clinic_efficacy_primary",
            "value": [1.0, 2.0],
            "direction": "beneficial",
        },
        bg={"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        clinical_design={
            "comparator_type": "placebo",
            "phase": "phase_3",
            "population": "adults",
            "study_id": "no_colon_bad_curie",
        },
    )
    result = validate_evidence(e)
    assert not result.ok
    assert any("study_id" in err for err in result.hard_errors)


def test_curie_valid_study_id_passes():
    """clinical_design.study_id with valid CURIE should pass."""
    e = _make_evidence(
        biological_level="rct",
        evidence_design="two_arm_comparison",
        xy={"count": 2},
        x={
            "type": "drug",
            "value": [
                {"entity_id": "chembl:CHEMBL25", "entity_label": "drug"},
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        y={
            "type": "test",
            "category": "clinic_efficacy_primary",
            "value": [1.0, 2.0],
            "direction": "beneficial",
        },
        bg={"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        clinical_design={
            "comparator_type": "placebo",
            "phase": "phase_3",
            "population": "adults",
            "study_id": "clinicaltrials:NCT01234567",
        },
    )
    result = validate_evidence(e)
    assert result.ok, result.hard_errors


def test_curie_non_id_curie_fields_validated():
    """model_organism, tissue, cell_type are type: curie but don't end in '_id'.
    They must be validated."""
    # model_organism without prefix separator -> reject
    e = _make_evidence()
    e["model_organism"] = "no_colon_bad_curie"
    result = validate_evidence(e)
    assert not result.ok
    assert any("model_organism" in err for err in result.hard_errors)

    # tissue without prefix separator -> reject
    e2 = _make_evidence()
    e2["tissue"] = "bad_tissue_no_prefix"
    result2 = validate_evidence(e2)
    assert not result2.ok
    assert any("tissue" in err for err in result2.hard_errors)

    # cell_type without prefix separator -> reject
    e3 = _make_evidence()
    e3["cell_type"] = "bad_cell_type"
    result3 = validate_evidence(e3)
    assert not result3.ok
    assert any("cell_type" in err for err in result3.hard_errors)


def test_curie_non_id_curie_fields_valid_passes():
    """model_organism, tissue, cell_type with valid CURIE should pass."""
    e = _make_evidence()
    e["model_organism"] = "NCBITaxon:9606"
    e["tissue"] = "uberon:0000955"
    e["cell_type"] = "cl:0000066"
    result = validate_evidence(e)
    assert result.ok, result.hard_errors
