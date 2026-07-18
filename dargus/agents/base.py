"""Base agent class and shared utilities."""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all Dargus agents."""

    name: str = "BaseAgent"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._load_default_config()
        self.skills: set[str] = set()

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        config_path = Path(__file__).resolve().parent.parent / "config" / "dargus_config.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    @abstractmethod
    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's task and return a structured result."""

    def register_skill(self, skill_path: str) -> None:
        """Register a SKILL extension."""
        self.skills.add(Path(skill_path).stem)
        logger.info("%s registered skill %s", self.name, skill_path)

    def list_skills(self) -> list[str]:
        """List registered skills."""
        return sorted(self.skills)

    def has_skill(self, skill_name: str) -> bool:
        """Check whether a skill is registered."""
        return skill_name in self.skills

    def _project_dir(self, project_id: str) -> Path:
        root = Path(self.config.get("projects", {}).get("root_dir", "projects"))
        return root / project_id

    def _outputs_dir(self, project_id: str, layer: str, task_name: str) -> Path:
        out = self._project_dir(project_id) / "outputs" / layer / task_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_five_pack(
        self,
        project_id: str,
        layer: str,
        task_name: str,
        report: str,
        figures: dict[str, bytes] | None,
        data: dict[str, Any] | None,
        code: str | None,
        embedding: dict[str, Any],
    ) -> dict[str, str]:
        """Write the standard five-pack output for a level agent."""
        out = self._outputs_dir(project_id, layer, task_name)

        report_path = out / "report.md"
        report_path.write_text(report, encoding="utf-8")

        fig_paths: dict[str, str] = {}
        if figures:
            fig_dir = out / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in figures.items():
                path = fig_dir / fname
                path.write_bytes(content)
                fig_paths[fname] = str(path)

        data_paths: dict[str, str] = {}
        if data:
            data_dir = out / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            for fname, obj in data.items():
                path = data_dir / fname
                if isinstance(obj, bytes):
                    path.write_bytes(obj)
                else:
                    path.write_text(_to_csv_text(obj), encoding="utf-8")
                data_paths[fname] = str(path)

        code_path = out / "code" / "analysis.py"
        if code:
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")

        embedding_path = out / "level_embedding.json"
        embedding_path.write_text(json.dumps(embedding, indent=2), encoding="utf-8")

        return {
            "report": str(report_path),
            "figures": fig_paths,
            "data": data_paths,
            "code": str(code_path) if code else "",
            "level_embedding": str(embedding_path),
        }

    def _trace(self, project_id: str, task_id: str, event: str, details: dict[str, Any]) -> None:
        trace_dir = self._project_dir(project_id) / "logs" / "agent_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{self.name}.jsonl"
        line = json.dumps(
            {
                "timestamp": _now_iso(),
                "task_id": task_id,
                "event": event,
                "details": details,
            },
            ensure_ascii=False,
        )
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _to_csv_text(obj: Any) -> str:
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj.to_csv(index=False)
    if isinstance(obj, list):
        return pd.DataFrame(obj).to_csv(index=False)
    raise TypeError(f"Cannot convert {type(obj)} to CSV text")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def new_task_id() -> str:
    return str(uuid.uuid4())
