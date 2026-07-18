from __future__ import annotations

from pathlib import Path
from typing import Any

from dargus import DirectorAgent


def _respond(success: bool, data: dict[str, Any] | None = None, error: str | None = None) -> dict:
    return {"success": success, "data": data or {}, "error": error}


def tool_status(project_id: str, projects_root: str = "projects") -> dict:
    root = Path(projects_root)
    project_dir = root / project_id
    if not project_dir.exists():
        return _respond(False, error=f"Project {project_id!r} not found at {project_dir}")

    director = DirectorAgent(config={"projects": {"root_dir": str(root)}})
    return _respond(True, data=director.status(project_id))
