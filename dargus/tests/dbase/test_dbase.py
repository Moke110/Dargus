"""Tests for DBase v0.15.5 — shard JSONL evidence store with three-axis records."""

import tempfile

from dargus.dbase import DBase
from dargus.dbase.validate import compute_evidence_id, validate_evidence


def _make_evidence(**overrides):
    """Return a valid three-axis evidence dict (descriptive = xy.count 0)."""
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 0},
        "x": {"type": "drug", "unit": None, "value": []},
        "y": {
            "type": "logP",
            "category": "pk_adme",
            "value": [3.5],
        },
        "bg": {"disease_id": [], "drugs": [], "genes": [], "model": None},
        "sources": [{"rank": 1, "type": "doi", "id": "10.1234/test"}],
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
    assert result.ok


def test_validate_evidence_rejects_empty_sources():
    result = validate_evidence(_make_evidence(sources=[]))
    assert not result.ok


def test_validate_evidence_rejects_invalid_level():
    result = validate_evidence(_make_evidence(biological_level="clinical"))
    assert not result.ok


def test_evidence_id_stable():
    """Evidence_id is content-addressed and stable (§5)."""
    e = {
        "biological_level": "rct",
        "evidence_design": "two_arm_comparison",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "unit": None,
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
        "bg": {"disease_id": ["mondo:0005180"], "drugs": [], "genes": [], "model": None},
        "clinical_design": {
            "comparator_type": "placebo",
            "phase": "phase_3",
            "population": "adults",
            "study_id": "clinicaltrials:NCT01234567",
        },
        "sources": [{"rank": 1, "type": "doi", "id": "10.1234/test"}],
    }
    assert compute_evidence_id(e) == compute_evidence_id(e)
    assert compute_evidence_id(e).startswith("ev_")


def test_evidence_id_distinct_for_different_y_type():
    e1 = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 0},
        "x": {"type": "drug", "value": []},
        "y": {"type": "logP", "category": "pk_adme", "value": [3.5]},
        "bg": {"disease_id": [], "drugs": [], "genes": [], "model": None},
        "sources": [{"rank": 1, "type": "doi", "id": "10.1234/test"}],
    }
    e2 = dict(e1)
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
            "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": [], "model": None},
            "clinical_design": {
                "comparator_type": "placebo",
                "phase": "phase_3",
                "population": "adults",
                "study_id": "clinicaltrials:NCT01234567",
            },
            "sources": [{"rank": 1, "type": "pmid", "id": "12345678"}],
        }
    )
    assert not result.ok
    assert any("control" in e.lower() for e in result.hard_errors)


def test_validate_sim_level_requires_provenance():
    """-sim levels must carry simulation_provenance."""
    e = _make_evidence(biological_level="molecular-sim")
    result = validate_evidence(e)
    assert not result.ok
    assert any("simulation_provenance" in e for e in result.hard_errors)
