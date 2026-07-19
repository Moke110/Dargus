"""Iris commander — project manager and prediction orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dargus.dbase import DBase, TemplateSchema
from dargus.dbase.manager import DBaseManager
from dargus.iris.ensemble import IrisEnsemble
from dargus.iris.selector import IrisSelector

logger = logging.getLogger(__name__)


class Iris:
    """Coordinates D-Base, expert system, and Iris-* agents for a project."""

    name = "Iris"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
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
        traces: dict[str, list[dict[str, Any]]] = {}
        if traces_dir.exists():
            for trace_file in traces_dir.glob("*.jsonl"):
                with trace_file.open("r", encoding="utf-8") as fh:
                    events = [json.loads(line) for line in fh if line.strip()]
                traces[trace_file.stem] = events
        status["agent_traces"] = traces
        status["outputs"] = self._list_outputs(project_dir)
        return status

    def ingest_project(self, project_id: str, datadir: str) -> dict[str, Any]:
        """Scan a local data directory and write records into D-Base."""
        from dargus.agents.reader import ReaderAgent

        dbase = DBase(project_id, root_dir=self.projects_root)
        self._ensure_default_templates(dbase)
        manager = DBaseManager(dbase)
        reader = ReaderAgent(self.config)
        scan = reader.scan_directory(datadir)
        instances: list[dict[str, Any]] = []
        for f in scan.get("data_files", []):
            instances.extend(reader.parse_data_file(f))

        for raw in instances:
            record = manager.fill_template(
                raw,
                source_metadata=raw.get("source", {"type": "user_upload"}),
            )
            manager.write_record(record)
        dbase.save()

        return {"project_id": project_id, "n_records": len(dbase.list_records())}

    def _ensure_default_templates(self, dbase: DBase) -> None:
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

    def plan_prediction(
        self,
        project_id: str,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Produce a prediction plan for review/confirmation."""
        if endpoints is None:
            endpoints = self._default_endpoints(disease_id)
        return {
            "project_id": project_id,
            "drug_ids": drug_ids,
            "disease_id": disease_id,
            "endpoints": endpoints,
            "agents": ["Iris-search", "Iris-analog", "Iris-bayes", "Iris-gnn", "Iris-llm"],
        }

    def predict(
        self,
        project_id: str,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str] | None = None,
        confirm_callback: Any = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run Iris-* agents and return an ensemble prediction.

        The optional ``confirm_callback`` receives the plan proposal and must
        return a truthy value before agents are executed.
        """
        plan = self.plan_prediction(project_id, drug_ids, disease_id, endpoints)
        if confirm_callback is not None and not confirm_callback(plan):
            raise RuntimeError("Prediction plan was not confirmed")

        dbase = DBase(project_id, root_dir=self.projects_root)
        selector = IrisSelector(dbase, config=self.config)
        return selector.predict(drug_ids, disease_id, endpoints=plan["endpoints"])

    def ensemble(
        self,
        predictions: dict[str, dict[str, dict[str, Any]]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Aggregate predictions from multiple Iris-* agents.

        ``weights`` maps agent names to positive weights. If omitted, inverse
        interval width weighting is used.
        """
        ensemble = IrisEnsemble()
        aggregated = ensemble.aggregate(predictions)
        if weights:
            for drug in aggregated:
                for endpoint in aggregated[drug]:
                    entry = aggregated[drug][endpoint]
                    components = entry.get("component_predictions", {})
                    weighted_low: list[float] = []
                    weighted_up: list[float] = []
                    denom = 0.0
                    for mode, interval in components.items():
                        w = weights.get(mode, 1.0)
                        weighted_low.append(interval["efficacy_low"] * w)
                        weighted_up.append(interval["efficacy_up"] * w)
                        denom += w
                    if denom > 0:
                        entry["efficacy_low"] = sum(weighted_low) / denom
                        entry["efficacy_up"] = sum(weighted_up) / denom
        return aggregated

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
