"""Tests for DBaseManager v1.0.0 — three-axis evidence dict API (50-field schema)."""

import tempfile

import pytest

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager


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


def _make_pairwise(**overrides):
    """Return a valid v1.0.0 three-axis pairwise evidence dict."""
    e = {
        "biological_level": "rct",
        "evidence_design": "two_arm_comparison",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {
                    "entity_id": "chembl:CHEMBL25",
                    "entity_label": "aspirin",
                    "dose": {"v": 100, "u": "mg"},
                },
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        "y": {
            "type": "HbA1c_change",
            "category": "clinic_efficacy_primary",
            "unit": "%",
            "value": [-0.8, 0.1],
            "dispersion": [
                {"type": "CI95", "value": [-1.2, -0.4]},
                {"type": "CI95", "value": [-0.1, 0.3]},
            ],
            "n_total": [500, 500],
            "direction": "beneficial",
            "effect": {"value": -0.9, "value_type": "MD"},
        },
        "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        "clinical_design": {
            "comparator_type": "placebo",
            "blinding": "double",
            "randomized": True,
            "phase": "phase_3",
            "n_arms": 2,
            "population": "adults",
            "study_id": "clinicaltrials:NCT01234567",
        },
        "sex": "mixed",
        "sources": [{"rank": 1, "type": "journal", "name": "PMID 34567890"}],
        "source_entry": "PMID:34567890",
        "source_time": "2025-11-20",
    }
    e.update(overrides)
    return e


def test_manager_write_record():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        result = manager.write_record(_make_evidence())
        assert result is True
        assert len(dbase.read_shards()) == 1


def test_manager_write_pairwise_record():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        result = manager.write_record(_make_pairwise())
        assert result is True
        record = dbase.read_shards()[0]
        assert record["evidence_id"].startswith("ev_")
        assert record["xy"]["count"] == 2


def test_manager_reset_clears_all_records():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        manager.write_record(_make_evidence())
        manager.reset()
        assert len(dbase.read_shards()) == 0


def test_manager_read_records():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        manager.write_record(_make_evidence())
        records = manager.read_records(y_type="logP")
        assert len(records) >= 1


def test_manager_build_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        record = manager.build_evidence(
            {
                "drug_id": "chembl:CHEMBL25",
                "readout_type": "ic50",
                "readout_category": "binding",
                "readout_value": 5.0,
                "readout_unit": "nM",
                "biological_level": "molecular",
            },
            source_metadata={
                "type": "journal",
                "name": "10.1234/test",
                "entry": "10.1234/test",
                "time": "2026-01-01",
            },
        )
        assert record["evidence_id"].startswith("ev_")
        assert "schema_version" not in record
        assert "x" in record
        assert "y" in record
        assert "bg" in record
        assert "xy" in record


def test_manager_write_record_rejects_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        with pytest.raises(ValueError):
            manager.write_record(_make_evidence(biological_level="invalid_level"))


def test_manager_read_records_by_filters():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        manager.write_record(_make_evidence())
        records = manager.read_records(y_type="logP")
        assert len(records) >= 1
        records = manager.read_records(x_entity="chembl:CHEMBL25")
        assert len(records) >= 1
        records = manager.read_records(level="molecular")
        assert len(records) >= 1


def test_manager_build_evidence_from_three_axis_raw():
    """build_evidence with three-axis raw input."""
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        record = manager.build_evidence(
            {
                "biological_level": "cellular",
                "xy": {"count": 1},
                "x": {
                    "type": "drug",
                    "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
                },
                "y": {
                    "type": "cell_viability",
                    "category": "viability",
                    "value": [0.5],
                    "unit": "fraction",
                },
                "bg": {"disease_id": [], "drugs": [], "genes": []},
            },
            source_metadata={
                "type": "journal",
                "name": "PMID 12345678",
                "entry": "PMID:12345678",
                "time": "2025-05-01",
            },
        )
        assert record["evidence_id"].startswith("ev_")
        assert record["y"]["type"] == "cell_viability"
        assert record["xy"]["count"] == 1


def test_manager_duplicate_detection():
    """Writing the same evidence twice returns False."""
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        evidence = _make_evidence()
        result1 = manager.write_record(evidence)
        assert result1 is True
        result2 = manager.write_record(evidence)
        assert result2 is False


def test_manager_read_record_by_id():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        evidence = _make_evidence()
        manager.write_record(evidence)
        eid = evidence["evidence_id"]
        found = manager.read_record(eid)
        assert found is not None
        assert found["evidence_id"] == eid


def test_compute_evidence_id_deterministic():
    """Same evidence computes the same id regardless of dict field ordering."""
    from dargus.dbase.validate import compute_evidence_id

    e1 = _make_evidence()
    e2 = dict(reversed(list(_make_evidence().items())))
    assert compute_evidence_id(e1) == compute_evidence_id(e2)


def test_compute_evidence_id_distinct():
    """Different y.type should yield different evidence_ids."""
    from dargus.dbase.validate import compute_evidence_id

    e1 = _make_evidence()
    e2 = {
        **_make_evidence(),
        "y": {**_make_evidence()["y"], "type": "clogP"},
    }
    assert compute_evidence_id(e1) != compute_evidence_id(e2)


def test_identity_includes_provenance():
    """source_entry / source_time changes alter the evidence_id."""
    from dargus.dbase.validate import compute_evidence_id

    e1 = _make_evidence()
    e2 = {**_make_evidence(), "source_entry": "10.1234/other"}
    e3 = {**_make_evidence(), "source_time": "2024-06-30"}
    assert compute_evidence_id(e1) != compute_evidence_id(e2)
    assert compute_evidence_id(e1) != compute_evidence_id(e3)


def test_identity_includes_bg_dose():
    """bg.dose_value participates in identity per design/2.1.1."""
    from dargus.dbase.validate import compute_evidence_id

    e1 = _make_evidence()
    e2 = _make_evidence()
    e2["bg"] = {**e2["bg"], "dose_value": 10.0, "dose_unit": "mg/kg"}
    assert compute_evidence_id(e1) != compute_evidence_id(e2)


def test_rct_sim_is_non_clinical():
    """rct-sim derives is_clinical=0 per design/2.1.2."""
    e = _make_pairwise(biological_level="rct-sim", clinical_design=None)
    del e["clinical_design"]
    from dargus.dbase.validate import validate_evidence

    result = validate_evidence(e)
    assert result.ok, result.hard_errors
    assert e["is_clinical"] == 0
