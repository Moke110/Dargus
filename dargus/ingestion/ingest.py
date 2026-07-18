from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dargus.dbase import DBase, TemplateSchema
from dargus.ingestion.converters.tdc_admet import TdcAdmetConverter
from dargus.ingestion.converters.tdc_dti import TdcDtiConverter
from dargus.temp_retriever import TempRetriever

CONVERTERS: dict[str, callable] = {
    "tier1_admet_solubility": lambda: TdcAdmetConverter("solubility"),
    "tier1_admet_lipophilicity": lambda: TdcAdmetConverter("lipophilicity"),
    "tier1_dti_davis": lambda: TdcDtiConverter("affinity"),
    "tier1_dti_kiba": lambda: TdcDtiConverter("affinity"),
}


def _register_templates(dbase: DBase) -> None:
    templates_dir = Path(__file__).parent.parent / "dbase" / "templates"
    for path in templates_dir.glob("*.yaml"):
        schema = TemplateSchema.from_yaml(path)
        if schema.template_id not in dbase._templates:
            dbase.add_template(schema)


def ingest_dataset(
    project_id: str,
    dataset_name: str,
    data_dir: str,
    projects_root: str = "projects",
) -> dict[str, Any]:
    if dataset_name not in CONVERTERS:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    dbase = DBase(project_id, root_dir=projects_root)
    _register_templates(dbase)

    converter_factory = CONVERTERS[dataset_name]
    converter = converter_factory()

    retriever = TempRetriever(dbase)
    n_added = 0
    for path in Path(data_dir).glob("*"):
        if not path.is_file():
            continue
        for row_idx, raw in enumerate(converter.convert(path)):
            record_id = hashlib.sha1(
                json.dumps(
                    {"path": path.name, "row": row_idx, "raw": raw},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            record = retriever.fill_template(
                raw,
                source_metadata={
                    "type": "public_db",
                    "database_id": dataset_name,
                    "record_id": record_id,
                    "source_file": path.name,
                    "source_row": row_idx,
                },
                suggested_template=converter.template_id,
            )
            # Ensure stable record_id
            record.record_id = f"{dataset_name}_{record_id}"
            retriever.write_record(record)
            n_added += 1

    dbase.save()
    return {"project_id": project_id, "dataset_name": dataset_name, "n_records": n_added}


def populate_project(
    project_id: str,
    dataset_names: list[str],
    data_root: str = "data/benchmarks",
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
