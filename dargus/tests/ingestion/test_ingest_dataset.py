"""Tests for ingest_dataset driving a registered converter end-to-end."""

from __future__ import annotations

import json

import pytest

from dargus.dbase import DBase
from dargus.ingestion.ingest import ingest_dataset


def _write_raw(tmp_path, raw_records):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    with (data_dir / "raw.jsonl").open("w", encoding="utf-8") as fh:
        for raw in raw_records:
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
    return data_dir


def _clinicaltrials_raw(nct: str, condition: str, drug: str = "Metformin") -> dict:
    return {
        "source": "clinicaltrials",
        "source_entry": f"clinicaltrials:{nct}",
        "source_time": "2026-08-03",
        "data": {
            "protocolSection": {
                "conditionsModule": {"conditions": [condition]},
                "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": drug}]},
                "designModule": {"phases": ["PHASE3"]},
                "outcomesModule": {"primaryOutcomes": [{"measure": "Primary endpoint"}]},
                "identificationModule": {"nctId": nct},
            }
        },
    }


def test_ingest_dataset_drives_registered_converter(tmp_path):
    """ingest_dataset reads raw.jsonl wrappers and produces D-Base records."""
    data_dir = _write_raw(
        tmp_path,
        [
            _clinicaltrials_raw("NCT00000001", "Essential Hypertension"),
            _clinicaltrials_raw("NCT00000002", "Sickle Cell Disease"),  # unmappable -> skip
        ],
    )
    projects_root = tmp_path / "projects"
    result = ingest_dataset(
        "test",
        "clinicaltrials",
        str(data_dir),
        projects_root=str(projects_root),
    )
    assert result["n_records"] == 1
    assert result["n_skipped"] == 1

    # the written record is validator-clean and in the project store
    dbase = DBase("test", root_dir=projects_root / "test")
    records = dbase.read_shards()
    assert len(records) == 1
    assert records[0]["bg"]["disease_id"] == ["mondo:0001134"]


def test_ingest_dataset_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        ingest_dataset("test", "not_a_dataset", str(tmp_path))
