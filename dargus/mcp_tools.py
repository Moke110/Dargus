from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dargus import DirectorAgent
from dargus.dbase import DBase, TemplateSchema
from dargus.temp_retriever import TempRetriever


def _respond(success: bool, data: dict[str, Any] | None = None, error: str | None = None) -> dict:
    return {"success": success, "data": data or {}, "error": error}


def tool_start_project(
    disease: str,
    target: str | None = None,
    endpoints: list[str] | None = None,
    data_paths: list[str] | None = None,
    projects_root: str = "projects",
) -> dict:
    try:
        director = DirectorAgent(config={"projects": {"root_dir": projects_root}})
        result = director.start_project(
            disease=disease,
            target=target,
            clinical_endpoints=endpoints,
            user_data_paths=data_paths,
        )
        return _respond(True, data=result)
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_ingest_data(
    project_id: str,
    datadir: str,
    projects_root: str = "projects",
) -> dict:
    try:
        from dargus.agents.reader import ReaderAgent

        director = DirectorAgent(config={"projects": {"root_dir": projects_root}})
        dbase = DBase(project_id, root_dir=projects_root)

        # Ensure default templates exist for common raw data shapes.
        _ensure_default_templates(dbase)

        reader = ReaderAgent(director.config)
        scan = reader.scan_directory(datadir)
        instances: list[dict] = []
        for f in scan.get("data_files", []):
            instances.extend(reader.parse_data_file(f))

        retriever = TempRetriever(dbase)
        for raw in instances:
            record = retriever.fill_template(
                raw,
                source_metadata=raw.get("source", {"type": "user_upload"}),
            )
            retriever.write_record(record)
        dbase.save()

        return _respond(
            True,
            data={"project_id": project_id, "n_records": len(dbase.list_records())},
        )
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def _ensure_default_templates(dbase: DBase) -> None:
    drug_vocab = "global_drug_vocab"
    disease_vocab = "global_disease_vocab"
    if "in_vitro_kinase_inhibition_v1" not in dbase._templates:
        dbase.add_template(
            TemplateSchema(
                template_id="in_vitro_kinase_inhibition_v1",
                fields=[
                    {"name": "biological_level", "type": "factor", "vocabulary": ["molecular"]},
                    {"name": "drug_id", "type": "factor", "vocabulary_ref": drug_vocab},
                    {"name": "disease_id", "type": "factor", "vocabulary_ref": disease_vocab},
                    {"name": "readout", "type": "float"},
                ],
            )
        )
    if "cell_viability_assay_v1" not in dbase._templates:
        dbase.add_template(
            TemplateSchema(
                template_id="cell_viability_assay_v1",
                fields=[
                    {"name": "biological_level", "type": "factor", "vocabulary": ["cellular"]},
                    {"name": "drug_id", "type": "factor", "vocabulary_ref": drug_vocab},
                    {"name": "disease_id", "type": "factor", "vocabulary_ref": disease_vocab},
                    {"name": "readout", "type": "float"},
                ],
            )
        )


def tool_search_literature(
    project_id: str,
    drug_ids: list[str],
    disease_id: str,
    projects_root: str = "projects",
) -> dict:
    try:
        from dargus.agents.report_searcher import ReportSearcher

        director = DirectorAgent(config={"projects": {"root_dir": projects_root}})
        searcher = ReportSearcher(director.config)
        result = searcher.search(drug_ids, disease_id)
        return _respond(True, data={"project_id": project_id, "result": result})
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_predict(
    project_id: str,
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
    projects_root: str = "projects",
) -> dict:
    try:
        from dargus.iris.selector import IrisSelector

        director = DirectorAgent(config={"projects": {"root_dir": projects_root}})
        dbase = DBase(project_id, root_dir=projects_root)
        selector = IrisSelector(dbase, config=director.config)
        predictions = selector.predict(drug_ids, disease_id, endpoints=endpoints)
        return _respond(True, data={"project_id": project_id, "predictions": predictions})
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_query_dbase(
    project_id: str,
    drug_id: str | None = None,
    disease_id: str | None = None,
    template_id: str | None = None,
    projects_root: str = "projects",
) -> dict:
    try:
        dbase = DBase(project_id, root_dir=projects_root)
        records = dbase.query(
            template_id=template_id,
            drug_id=drug_id,
            disease_id=disease_id,
        )
        return _respond(
            True,
            data={
                "project_id": project_id,
                "n_records": len(records),
                "records": [json.loads(r.model_dump_json()) for r in records],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_status(project_id: str, projects_root: str = "projects") -> dict:
    root = Path(projects_root)
    project_dir = root / project_id
    if not project_dir.exists():
        return _respond(False, error=f"Project {project_id!r} not found at {project_dir}")

    director = DirectorAgent(config={"projects": {"root_dir": str(root)}})
    return _respond(True, data=director.status(project_id))
