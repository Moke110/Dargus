"""Tests for the ingest converter framework (pipeline.py) and golden re-run driver.

The golden slice data itself is gitignored, so these tests exercise the
framework with a small synthetic slice: a converter produces evidence, the
pipeline validates + writes per-level files, and skips are manifested with no
truncation.
"""

from __future__ import annotations

import json
from pathlib import Path

from dargus.dbase.validate import validate_evidence
from dargus.ingestion.converters.base import BaseConverter
from dargus.ingestion.converters.pipeline import (
    SkipRecord,
    convert_slice,
    write_manifest,
)


class _FakeClinicalConverter(BaseConverter):
    """A minimal converter: one valid rct record, one skip, per raw line."""

    template_id = "fakeclinical"

    def convert(self, raw: dict) -> list[dict | SkipRecord]:
        se = str(raw.get("source_entry", ""))
        if se.endswith("BAD"):
            return [
                SkipRecord(
                    source_entry=se,
                    source=self.template_id,
                    reason="unmapped_disease",
                    detail="x",
                )
            ]
        return [
            {
                "biological_level": "rct",
                "evidence_design": "descriptive",
                "xy": {"count": 1},
                "x": {
                    "type": "drug",
                    "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
                },
                "y": {"type": "endpoint", "category": "clinic_efficacy_primary", "value": [1.0]},
                "bg": {"disease_id": ["mondo:0001134"], "drugs": [], "genes": []},
                "clinical_design": {
                    "comparator_type": "no_treatment",
                    "population": "adults",
                    "study_id": "clinicaltrials:NCT00000001",
                },
                "source_entry": se,
                "source_time": str(raw.get("source_time", "")),
            }
        ]


def _write_raw(tmp_path: Path, records: list[dict]) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "raw.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return raw_dir


def test_convert_slice_writes_validated_evidence_and_skips(tmp_path):
    raw_dir = _write_raw(
        tmp_path,
        [
            {
                "source": "fakeclinical",
                "source_entry": "NCT00000001",
                "source_time": "2026-01-01",
                "data": {},
            },
            {
                "source": "fakeclinical",
                "source_entry": "NCT00000002",
                "source_time": "2026-01-01",
                "data": {},
            },
            {
                "source": "fakeclinical",
                "source_entry": "NCT_BAD",
                "source_time": "2026-01-01",
                "data": {},
            },
        ],
    )
    out_dir = tmp_path / "out"
    summary = convert_slice(
        _FakeClinicalConverter(),
        raw_dir=raw_dir,
        out_dir=out_dir,
        project_id="test",
    )
    assert summary["n_records"] == 2
    assert summary["n_skipped"] == 1

    # evidence is validated + provenance-injected (sources present)
    records = [json.loads(line) for line in (out_dir / "rct.jsonl").open()]
    assert len(records) == 2
    for rec in records:
        assert validate_evidence(rec).ok
        assert rec["sources"] == [{"rank": 1, "type": "database", "name": "fakeclinical"}]
        assert rec["source_entry"] in ("NCT00000001", "NCT00000002")

    # skips manifested completely (no truncation)
    skips = [json.loads(line) for line in (out_dir / "skips.jsonl").open()]
    assert len(skips) == 1
    assert skips[0]["reason"] == "unmapped_disease"
    assert skips[0]["source_entry"] == "NCT_BAD"


def test_convert_slice_dedups_identical_rows(tmp_path):
    """Two raw rows producing identical evidence collapse to one record."""
    raw_dir = _write_raw(
        tmp_path,
        [
            {
                "source": "fakeclinical",
                "source_entry": "NCT00000001",
                "source_time": "2026-01-01",
                "data": {},
            },
            {
                "source": "fakeclinical",
                "source_entry": "NCT00000001",
                "source_time": "2026-01-01",
                "data": {},
            },
        ],
    )
    out_dir = tmp_path / "out"
    convert_slice(_FakeClinicalConverter(), raw_dir=raw_dir, out_dir=out_dir, project_id="test")
    records = [json.loads(line) for line in (out_dir / "rct.jsonl").open()]
    assert len(records) == 1


def test_write_manifest_no_truncation(tmp_path):
    skips = [
        SkipRecord(source_entry=f"NCT{i}", source="clinicaltrials", reason="unmapped_disease")
        for i in range(120)  # exceeds the prior 50-entry truncation
    ]
    path = tmp_path / "skips.jsonl"
    write_manifest(skips, path)
    lines = [line for line in path.open() if line.strip()]
    assert len(lines) == 120
