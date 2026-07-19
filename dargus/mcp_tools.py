from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.iris.commander import Iris


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
        iris = Iris(config={"projects": {"root_dir": projects_root}})
        result = iris.start_project(
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
        iris = Iris(config={"projects": {"root_dir": projects_root}})
        result = iris.ingest_project(project_id, datadir)
        return _respond(True, data=result)
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_search_literature(
    project_id: str,
    drug_ids: list[str],
    disease_id: str,
    projects_root: str = "projects",
) -> dict:
    try:
        from dargus.agents.report_searcher import ReportSearcher

        iris = Iris(config={"projects": {"root_dir": projects_root}})
        searcher = ReportSearcher(iris.config)
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
        iris = Iris(config={"projects": {"root_dir": projects_root}})
        predictions = iris.predict(
            project_id=project_id,
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints,
        )
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

    iris = Iris(config={"projects": {"root_dir": str(root)}})
    return _respond(True, data=iris.status(project_id))
