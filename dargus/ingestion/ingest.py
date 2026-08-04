"""Ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.dbase.store import DBaseStore
from dargus.ingestion.converters.base import SkipRecord
from dargus.ingestion.converters.clinicaltrials import ClinicalTrialsConverter
from dargus.ingestion.converters.clinvar import ClinVarConverter
from dargus.ingestion.converters.openfda import OpenFDAConverter

CONVERTERS: dict[str, callable] = {
    "clinicaltrials": ClinicalTrialsConverter,
    "openfda": OpenFDAConverter,
    "clinvar": ClinVarConverter,
}


def ingest_dataset(
    project_id: str,
    dataset_name: str,
    data_dir: str,
    projects_root: str = "projects",
) -> dict[str, Any]:
    """Drive one registered converter over a source's ``raw.jsonl`` wrappers.

    Reads every provenance wrapper under *data_dir*, passes it to the
    converter, runs each evidence dict through ``DBaseStore.build_evidence`` +
    ``write_record`` (dedup by content-addressed ``evidence_id``), and
    reports skips as a summary count.
    """
    if dataset_name not in CONVERTERS:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    dbase = DBase(project_id, root_dir=Path(projects_root) / project_id)
    converter = CONVERTERS[dataset_name]()

    manager = DBaseStore(dbase)
    n_added = 0
    n_skipped = 0
    for path in sorted(Path(data_dir).glob("raw.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for raw in _iter_raw_lines(fh):
                for item in converter.convert(raw):
                    if isinstance(item, SkipRecord):
                        n_skipped += 1
                        continue
                    record = manager.build_evidence(
                        item,
                        source_metadata={
                            "type": "database",
                            "name": dataset_name,
                            "entry": str(raw.get("source_entry", "")),
                            "time": str(raw.get("source_time", "")),
                        },
                    )
                    if manager.write_record(record) is True:
                        n_added += 1

    return {
        "project_id": project_id,
        "dataset_name": dataset_name,
        "n_records": n_added,
        "n_skipped": n_skipped,
    }


def _iter_raw_lines(fh):
    import json as _json

    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            yield _json.loads(line)
        except _json.JSONDecodeError:
            continue


def populate_project(
    project_id: str,
    dataset_names: list[str],
    data_root: str = "data",
    projects_root: str = "projects",
) -> dict[str, Any]:
    totals = {}
    for name in dataset_names:
        parts = name.split("_")
        tier = parts[0]
        category = parts[1]
        data_dir = Path(data_root) / tier / category
        result = ingest_dataset(project_id, name, str(data_dir), projects_root)
        totals[name] = result["n_records"]
    return {"project_id": project_id, "totals": totals}
