"""Tests for the ClinVar converter (ingestion.converters.clinvar)."""

from __future__ import annotations

from dargus.dbase import DBase, DBaseStore
from dargus.dbase.validate import validate_evidence
from dargus.ingestion.converters.clinvar import ClinVarConverter
from dargus.ingestion.converters.pipeline import SkipRecord


def _variant(uid, title, trait_name="", trait_xrefs=None, gene="CACNG2"):
    traits = []
    if trait_name or trait_xrefs:
        trait = {}
        if trait_name:
            trait["trait_name"] = trait_name
        if trait_xrefs:
            trait["trait_xrefs"] = trait_xrefs
        traits.append(trait)
    return {
        "uid": uid,
        "accession": f"VCV{uid}",
        "title": title,
        "genes": [{"symbol": gene, "geneid": "10369"}],
        "germline_classification": {"trait_set": traits},
    }


def _wrapper(source_entry: str, variants: list[dict]) -> dict:
    result_map = {"uids": [str(v["uid"]) for v in variants]}
    for v in variants:
        result_map[str(v["uid"])] = v
    return {
        "source": "clinvar",
        "source_entry": source_entry,
        "source_time": "2026-08-04",
        "data": {
            "formatted_results": {"result": result_map},
        },
    }


def _build(raw_output: dict) -> dict:
    dbase = DBase("test", root_dir="/tmp/dargus-test-store")
    manager = DBaseStore(dbase)
    return manager.build_evidence(
        raw_output,
        source_metadata={
            "type": "database",
            "name": "clinvar",
            "entry": raw_output["source_entry"],
            "time": raw_output["source_time"],
        },
    )


def test_variant_with_mondo_xref_produces_epi():
    converter = ClinVarConverter()
    raw = _wrapper(
        "target:CYP17A1",
        [
            _variant(
                "90793",
                "NM_000102.4(CYP17A1):c.985C>A",
                trait_name="Deficiency of steroid 17-alpha-monooxygenase",
                trait_xrefs=[{"db_source": "MONDO", "db_id": "MONDO:0008730"}],
                gene="CYP17A1",
            )
        ],
    )
    items = converter.convert(raw)
    assert len(items) == 1
    rec = items[0]
    assert not isinstance(rec, SkipRecord)
    assert rec["biological_level"] == "epi"
    assert "mondo:0008730" in rec["bg"]["disease_id"]
    # gene entity is label-only (no invented CURIE)
    assert rec["bg"]["genes"][0]["entity_label"] == "CYP17A1"
    assert rec["bg"]["genes"][0]["entity_id"] is None
    built = _build(rec)
    assert validate_evidence(built).ok


def test_variant_resolves_via_resolver_fallback():
    converter = ClinVarConverter()
    raw = _wrapper(
        "target:CACNG2",
        [_variant("123", "NM_006078.5(CACNG2):c.212-17C>G", trait_name="Epilepsy")],
    )
    rec = converter.convert(raw)[0]
    assert "mondo:0005027" in rec["bg"]["disease_id"]  # epilepsy


def test_unmappable_variant_skips():
    converter = ClinVarConverter()
    raw = _wrapper(
        "target:CACNG2",
        [_variant("456", "NM_006078.5(CACNG2):c.1A>G", trait_name="not specified")],
    )
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "unmapped_disease"


def test_multi_variant_wrapper_skips_unmappable_keeps_mappable():
    converter = ClinVarConverter()
    raw = _wrapper(
        "target:CACNG2",
        [
            _variant("1", "NM_006078.5(CACNG2):c.2A>G", trait_name="not specified"),
            _variant("2", "NM_006078.5(CACNG2):c.3A>G", trait_name="Epilepsy"),
        ],
    )
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)  # variant 1 skipped
    assert not isinstance(items[1], SkipRecord)  # variant 2 kept
    assert "mondo:0005027" in items[1]["bg"]["disease_id"]


def test_variant_source_entry_time_preserved():
    converter = ClinVarConverter()
    raw = _wrapper(
        "target:CACNG2",
        [_variant("7", "NM_006078.5(CACNG2):c.4A>G", trait_name="Epilepsy")],
    )
    rec = converter.convert(raw)[0]
    assert rec["source_entry"] == "target:CACNG2"
    assert rec["source_time"] == "2026-08-04"
