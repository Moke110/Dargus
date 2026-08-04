"""Tests for the OpenFDA converter (ingestion.converters.openfda)."""

from __future__ import annotations

from dargus.dbase import DBase, DBaseStore
from dargus.dbase.validate import validate_evidence
from dargus.ingestion.converters.openfda import OpenFDAConverter
from dargus.ingestion.converters.pipeline import SkipRecord


def _wrapper(drug: str, indications: str = "", status: str = "") -> dict:
    data = {"drug": drug}
    if indications:
        data["indications"] = indications
    if status:
        data["status"] = status
    return {
        "source": "openfda",
        "source_entry": f"drug:{drug}",
        "source_time": "2026-08-04",
        "data": data,
    }


def _build(raw_output: dict) -> dict:
    dbase = DBase("test", root_dir="/tmp/dargus-test-store")
    manager = DBaseStore(dbase)
    return manager.build_evidence(
        raw_output,
        source_metadata={
            "type": "database",
            "name": "openfda",
            "entry": raw_output["source_entry"],
            "time": raw_output["source_time"],
        },
    )


def test_resolvable_indication_produces_epi_evidence():
    converter = OpenFDAConverter()
    raw = _wrapper(
        "gemcitabine",
        "1 INDICATIONS AND USAGE Gemcitabine Injection is indicated for treatment of cancer",
    )
    items = converter.convert(raw)
    assert len(items) == 1
    rec = items[0]
    assert not isinstance(rec, SkipRecord)
    assert rec["biological_level"] == "epi"
    assert rec["bg"]["disease_id"], "disease_id must be populated"
    # provenance preserved
    assert rec["source_entry"] == "drug:gemcitabine"
    assert rec["source_time"] == "2026-08-04"
    built = _build(rec)
    assert validate_evidence(built).ok


def test_drug_resolves_to_chembl():
    converter = OpenFDAConverter()
    raw = _wrapper("metformin", "indicated for diabetes mellitus")
    rec = converter.convert(raw)[0]
    assert rec["x"]["value"][0]["entity_id"] == "chembl:CHEMBL1431"


def test_unresolvable_indication_skips_with_reason():
    converter = OpenFDAConverter()
    raw = _wrapper("some_drug", "indicated for a rare undiagnosable syndrome of unknown origin")
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "unmapped_disease"


def test_404_status_skips():
    converter = OpenFDAConverter()
    raw = _wrapper("tegafur", status="HTTP 404")
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "no_fda_label"


def test_missing_indications_skips():
    converter = OpenFDAConverter()
    raw = _wrapper("some_drug")
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "no_indications"


def test_unresolvable_drug_uses_entity_label_only():
    converter = OpenFDAConverter()
    raw = _wrapper("NotARealDrugXYZ", "indicated for diabetes mellitus")
    rec = converter.convert(raw)[0]
    assert rec["x"]["value"][0]["entity_id"] is None
    assert rec["x"]["value"][0]["entity_label"] == "NotARealDrugXYZ"
