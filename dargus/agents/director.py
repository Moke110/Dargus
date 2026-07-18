"""DirectorAgent — project manager and workflow orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dargus.agents.base import BaseAgent, new_task_id
from dargus.dbase import DBase

logger = logging.getLogger(__name__)


class TaskPool:
    """Simple rolling task pool with dependency tracking."""

    def __init__(self):
        self._pending: list[dict] = []
        self._completed: set[str] = set()

    def add(self, task: dict) -> None:
        self._pending.append(task)

    def ready(self) -> list[dict]:
        return [
            t
            for t in self._pending
            if t["id"] not in self._completed
            and all(dep in self._completed for dep in t.get("deps", []))
        ]

    def complete(self, task_id: str, spawn: list[dict] | None = None) -> None:
        self._completed.add(task_id)
        for t in spawn or []:
            self.add(t)

    def is_done(self) -> bool:
        return all(t["id"] in self._completed for t in self._pending)


class DirectorAgent(BaseAgent):
    """Creates projects, dispatches tasks, tracks progress, aggregates outputs."""

    name = "DirectorAgent"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.projects_root = Path(self.config.get("projects", {}).get("root_dir", "projects"))

    def start_project(
        self,
        disease: str,
        target: str | None = None,
        clinical_endpoints: list[str] | None = None,
        user_data_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new project directory, config, and database."""
        project_id = self._make_project_id(disease, target)
        project_dir = self.projects_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in [
            "literature/molecular",
            "literature/cellular",
            "literature/exvivo",
            "literature/animal",
            "literature/clinical",
            "literature/epidemiology",
            "literature/translation",
            "outputs/molecular",
            "outputs/cellular",
            "outputs/exvivo",
            "outputs/animal",
            "outputs/clinical",
            "outputs/epidemiology",
            "translation",
            "synthesis",
            "logs/agent_traces",
        ]:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Write project config
        if clinical_endpoints is None:
            clinical_endpoints = self._default_endpoints(disease)
        project_config = {
            "project_id": project_id,
            "disease": disease,
            "target": target,
            "clinical_endpoints": clinical_endpoints,
            "user_data_paths": user_data_paths or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        config_path = project_dir / "project_config.yaml"
        config_path.write_text(yaml.safe_dump(project_config), encoding="utf-8")

        # Initialize D-Base
        DBase(project_id, root_dir=str(self.projects_root))

        logger.info("Started project %s at %s", project_id, project_dir)
        return {"project_id": project_id, "project_dir": str(project_dir)}

    def status(self, project_id: str) -> dict[str, Any]:
        """Read project status from config and agent traces."""
        project_dir = self.projects_root / project_id
        config_path = project_dir / "project_config.yaml"
        status: dict[str, Any] = {"project_id": project_id}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                status["config"] = yaml.safe_load(fh)
        traces_dir = project_dir / "logs" / "agent_traces"
        traces = {}
        if traces_dir.exists():
            for trace_file in traces_dir.glob("*.jsonl"):
                with trace_file.open("r", encoding="utf-8") as fh:
                    events = [json.loads(line) for line in fh if line.strip()]
                traces[trace_file.stem] = events
        status["agent_traces"] = traces
        status["outputs"] = self._list_outputs(project_dir)
        return status

    def run_workflow(self, workflow_name: str, project_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run a pre-defined workflow."""
        if workflow_name == "target_efficacy_scan":
            from dargus.workflows.target_efficacy_scan import run

            return run(project_id, director=self, **kwargs)
        raise ValueError(f"Unknown workflow: {workflow_name}")

    def run_workflow_v4(
        self,
        workflow_name: str,
        project_id: str,
        drug_ids: list[str],
        disease_id: str,
        datadir: str | None = None,
    ) -> dict[str, Any]:
        """Rolling-schedule MVP workflow for D-Base."""
        from dargus.dbase import DBase

        dbase = DBase(project_id, root_dir=self.projects_root)

        pool = TaskPool()
        pool.add({"id": "scan_local", "type": "reader_scan", "deps": []})
        pool.add({"id": "search_web", "type": "report_search", "deps": []})

        report_search_result: dict[str, Any] | None = None
        iris_predictions: dict[str, Any] | None = None
        while not pool.is_done():
            ready = pool.ready()
            if not ready:
                break
            for task in ready:
                result = self._execute_task(
                    task,
                    project_id=project_id,
                    drug_ids=drug_ids,
                    disease_id=disease_id,
                    datadir=datadir,
                    dbase=dbase,
                )
                if task["type"] == "report_search":
                    report_search_result = result.get("result")
                if task["type"] == "iris_predict":
                    iris_predictions = result.get("result")
                pool.complete(task["id"], spawn=result.get("spawn", []))

        return {
            "project_id": project_id,
            "dbase_ready": True,
            "n_records": len(dbase.list_records()),
            "report_search_result": report_search_result,
            "predictions": iris_predictions or {},
        }

    def _execute_task(
        self,
        task: dict[str, Any],
        project_id: str,
        drug_ids: list[str],
        disease_id: str,
        datadir: str | None,
        dbase: Any,
    ) -> dict[str, Any]:
        from dargus.agents.reader import ReaderAgent
        from dargus.agents.report_searcher import ReportSearcher
        from dargus.temp_retriever import TempRetriever

        task_type = task["type"]
        if task_type == "iris_predict":
            from dargus.dbase import DBase
            from dargus.iris.selector import IrisSelector

            selector = IrisSelector(DBase(project_id, root_dir=self.projects_root))
            return {"result": selector.predict(drug_ids, disease_id)}

        if task_type == "reader_scan":
            if not datadir:
                return {"spawn": []}
            reader = ReaderAgent(self.config)
            scan = reader.scan_directory(datadir)
            instances: list[dict] = []
            for f in scan["data_files"]:
                instances.extend(reader.parse_data_file(f))
            return {
                "spawn": [
                    {
                        "id": "ingest",
                        "type": "temp_retriever_ingest",
                        "deps": ["scan_local"],
                        "payload": instances,
                    }
                ]
            }

        if task_type == "report_search":
            searcher = ReportSearcher(self.config)
            return {"result": searcher.search(drug_ids, disease_id)}

        if task_type == "temp_retriever_ingest":
            payload = task.get("payload", [])
            retriever = TempRetriever(dbase)
            for raw in payload:
                record = retriever.fill_template(
                    raw,
                    source_metadata=raw.get("source", {"type": "user_upload"}),
                )
                retriever.write_record(record)
            dbase.save()
            return {
                "spawn": [
                    {
                        "id": "predict",
                        "type": "iris_predict",
                        "deps": ["ingest", "search_web"],
                    }
                ]
            }

        if task_type == "iris_predict":
            return {"spawn": []}

        return {"spawn": []}

    def assign_task(self, agent_name: str, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a task to an agent class (in-process)."""
        task_id = new_task_id()
        task_spec["task_id"] = task_id
        agent = self._load_agent(agent_name)
        logger.info("Director assigning task %s to %s", task_id, agent_name)
        return agent.run(task_spec)

    def _load_agent(self, agent_name: str) -> BaseAgent:
        mapping = {
            "MoleculeAgent": "dargus.agents.molecular.MoleculeAgent",
            "EpiAgent": "dargus.agents.epidemiology.EpiAgent",
            "RetrieverAgent": "dargus.agents.retriever.RetrieverAgent",
            "DataMaster": "dargus.database.DataMaster",
            "TranslateAgent": "dargus.agents.translate.TranslateAgent",
        }
        if agent_name not in mapping:
            raise ValueError(f"Unknown agent: {agent_name}")
        module_path, class_name = mapping[agent_name].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(config=self.config)

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute a director-level task (delegation or workflow)."""
        task_type = task_spec.get("task_type")
        if task_type == "run_workflow":
            return self.run_workflow(
                task_spec["task_spec"]["workflow_name"],
                task_spec["project_id"],
            )
        if task_type == "start_project":
            spec = task_spec["task_spec"]
            return self.start_project(
                disease=spec["disease"],
                target=spec.get("target"),
                clinical_endpoints=spec.get("clinical_endpoints"),
                user_data_paths=spec.get("user_data_paths"),
            )
        return {"status": "error", "message": f"Unknown director task type: {task_type}"}

    def _make_project_id(self, disease: str, target: str | None) -> str:
        safe_disease = disease.replace(" ", "_").replace("'", "")
        safe_target = (target or "unknown").replace(" ", "_")
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{safe_target}_{safe_disease}_{date}"

    def _default_endpoints(self, disease: str) -> list[str]:
        return (
            self.config.get("projects", {})
            .get("default_endpoints", {})
            .get(disease, ["primary_endpoint_change"])
        )

    def _list_outputs(self, project_dir: Path) -> dict[str, list[str]]:
        outputs_dir = project_dir / "outputs"
        result: dict[str, list[str]] = {}
        if not outputs_dir.exists():
            return result
        for layer_dir in outputs_dir.iterdir():
            if layer_dir.is_dir():
                result[layer_dir.name] = [str(p) for p in layer_dir.rglob("*") if p.is_file()]
        return result
