"""Tests for the ClinicalTrials converter (ingestion.converters.clinicaltrials)."""

from __future__ import annotations

from dargus.dbase import DBase, DBaseStore
from dargus.dbase.validate import validate_evidence
from dargus.ingestion.converters.clinicaltrials import ClinicalTrialsConverter
from dargus.ingestion.converters.pipeline import SkipRecord


def _build(raw_output: dict) -> dict:
    """Run one converter output through the canonical build_evidence seam."""
    dbase = DBase("test", root_dir="/tmp/dargus-test-store")
    manager = DBaseStore(dbase)
    return manager.build_evidence(
        raw_output,
        source_metadata={
            "type": "database",
            "name": "clinicaltrials",
            "entry": raw_output["source_entry"],
            "time": raw_output["source_time"],
        },
    )


def _wrapper(nct: str, conditions: list[str], interventions: list[dict], **extra) -> dict:
    return {
        "source": "clinicaltrials",
        "source_entry": f"clinicaltrials:{nct}",
        "source_time": "2026-08-03",
        "data": {
            "protocolSection": {
                "identificationModule": {"nctId": nct},
                "designModule": {"phases": ["PHASE3"]},
                "armsInterventionsModule": {"interventions": interventions},
                "conditionsModule": {"conditions": conditions},
                "outcomesModule": {"primaryOutcomes": [{"measure": "Primary endpoint measure"}]},
                **extra,
            }
        },
    }


def test_resolvable_condition_produces_rct_evidence():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00938262",
        conditions=["Essential Hypertension"],
        interventions=[{"type": "DRUG", "name": "Fimasartan"}],
    )
    items = converter.convert(raw)
    assert len(items) == 1
    rec = items[0]
    assert not isinstance(rec, SkipRecord)
    assert rec["biological_level"] == "rct"
    assert rec["bg"]["disease_id"] == ["mondo:0001134"]
    assert rec["clinical_design"]["study_id"] == "clinicaltrials:NCT00938262"
    # source_entry/time carried verbatim from the wrapper
    assert rec["source_entry"] == "clinicaltrials:NCT00938262"
    assert rec["source_time"] == "2026-08-03"
    # passes the hard validator through the canonical build_evidence seam
    built = _build(rec)
    result = validate_evidence(built)
    assert result.ok, result.hard_errors


def test_multi_condition_unions_disease_ids():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000001",
        conditions=["Essential Hypertension", "Schizophrenia"],
        interventions=[{"type": "DRUG", "name": "Metformin"}],
    )
    rec = converter.convert(raw)[0]
    assert "mondo:0001134" in rec["bg"]["disease_id"]
    assert "mondo:0005090" in rec["bg"]["disease_id"]


def test_unmappable_condition_skips_with_reason():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000002",
        conditions=["Sickle Cell Disease"],
        interventions=[{"type": "DRUG", "name": "Hydroxyurea"}],
    )
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "unmapped_disease"


def test_partial_mapping_skips_with_reason():
    """If at least one condition resolves, we keep; if none resolve, skip."""
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000003",
        conditions=["Sickle Cell Disease", "Essential Hypertension"],
        interventions=[{"type": "DRUG", "name": "Drug X"}],
    )
    rec = converter.convert(raw)[0]
    assert rec["bg"]["disease_id"] == ["mondo:0001134"]


def test_no_drug_intervention_skips():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000004",
        conditions=["Essential Hypertension"],
        interventions=[{"type": "BEHAVIORAL", "name": "Diet"}],
    )
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "no_drug_intervention"


def test_no_condition_skips():
    converter = ClinicalTrialsConverter()
    raw = _wrapper("NCT00000005", conditions=[], interventions=[{"type": "DRUG", "name": "X"}])
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "no_condition"


def test_non_nct_source_entry_skips():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000006",
        conditions=["Essential Hypertension"],
        interventions=[{"type": "DRUG", "name": "X"}],
    )
    raw["source_entry"] = "clinicaltrials:no_results"
    items = converter.convert(raw)
    assert isinstance(items[0], SkipRecord)
    assert items[0].reason == "malformed_record"


def test_drug_resolves_to_chembl():
    converter = ClinicalTrialsConverter()
    raw = _wrapper(
        "NCT00000007",
        conditions=["Essential Hypertension"],
        interventions=[{"type": "DRUG", "name": "Metformin"}],
    )
    rec = converter.convert(raw)[0]
    assert rec["x"]["value"][0]["entity_id"] == "chembl:CHEMBL1431"
