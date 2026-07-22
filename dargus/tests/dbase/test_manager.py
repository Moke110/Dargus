"""Tests for DBaseManager v0.15.0 — evidence dict API."""

import tempfile

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager


def _make_evidence(**overrides):
    e = {
        "biological_level": "molecular",
        "evidence_design": "single_arm",
        "readout_type": "ic50",
        "readout_category": "binding",
        "readout_value": 5.0,
        "readout_unit": "nM",
        "interventions": [
            {"role": "primary", "entity_type": "small_molecule", "entity_id": "chembl:CHEMBL25"}
        ],
        "sources": [{"rank": 1, "type": "doi", "id": "10.1234/test"}],
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
        records = manager.read_records(readout_type="ic50")
        assert len(records) >= 1


def test_manager_build_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        record = manager.build_evidence(
            {
                "drug_id": "aspirin",
                "readout_type": "ic50",
                "readout_category": "binding",
                "readout_value": 5.0,
                "readout_unit": "nM",
                "biological_level": "molecular",
            },
            source_metadata={"type": "doi", "id": "10.1234/test"},
        )
        assert record["evidence_id"].startswith("ev_")
        assert record["schema_version"] == "v0.15.0"


def test_manager_write_record_rejects_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        try:
            manager.write_record(_make_evidence(biological_level="invalid_level"))
            assert False, "should have raised"
        except ValueError:
            pass


def test_manager_read_records_with_compat_aliases():
    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase("test", root_dir=tmp)
        manager = DBaseManager(dbase)
        manager.write_record(_make_evidence())
        records = manager.read_records(template_id="ic50")
        assert len(records) >= 1
