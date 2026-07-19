from __future__ import annotations

import json
import os
from typing import Any

from dargus.dbase import DBase
from dargus.iris.commander import Iris


def _respond(success: bool, data: dict[str, Any] | None = None, error: str | None = None) -> dict:
    return {"success": success, "data": data or {}, "error": error}


def _set_dargus_home(projects_root: str) -> None:
    """Temporarily override DARGUS_HOME for the global D-Base."""
    os.environ["DARGUS_HOME"] = projects_root


def tool_start_project(
    disease: str,
    target: str | None = None,
    endpoints: list[str] | None = None,
    data_paths: list[str] | None = None,
    projects_root: str = "projects",
) -> dict:
    try:
        _set_dargus_home(projects_root)
        iris = Iris()
        # In the new global D-Base model, "starting a project" means ensuring
        # the D-Base is initialized and reporting its status.
        status = iris.status()
        status["disease"] = disease
        status["target"] = target
        status["endpoints"] = endpoints or ["primary_endpoint_change"]
        return _respond(True, data=status)
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_ingest_data(
    project_id: str,
    datadir: str,
    projects_root: str = "projects",
) -> dict:
    try:
        _set_dargus_home(projects_root)
        iris = Iris()
        report = iris.train(datadir)
        return _respond(
            True,
            data={
                "n_records": report.n_records,
                "dbase_size": report.dbase_size,
                "errors": report.errors,
            },
        )
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

        _set_dargus_home(projects_root)
        iris = Iris()
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
        _set_dargus_home(projects_root)
        iris = Iris()
        predictions = iris.infer(
            drug_ids=drug_ids,
            disease_id=disease_id,
            endpoints=endpoints,
        )
        return _respond(True, data={"predictions": predictions})
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
        _set_dargus_home(projects_root)
        dbase = DBase.global_instance()
        records = dbase.query(
            template_id=template_id,
            drug_id=drug_id,
            disease_id=disease_id,
        )
        return _respond(
            True,
            data={
                "n_records": len(records),
                "records": [json.loads(r.model_dump_json()) for r in records],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))


def tool_status(project_id: str, projects_root: str = "projects") -> dict:
    try:
        _set_dargus_home(projects_root)
        iris = Iris()
        status = iris.status()
        return _respond(True, data=status)
    except Exception as exc:  # noqa: BLE001
        return _respond(False, error=str(exc))
