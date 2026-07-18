from __future__ import annotations

from typing import Any


def ingest_dataset(
    project_id: str,
    dataset_name: str,
    data_dir: str,
    projects_root: str = "projects",
) -> dict[str, Any]:
    raise NotImplementedError("ingest_dataset will be implemented in Task 5")


def populate_project(
    project_id: str,
    dataset_names: list[str],
    data_root: str = "data/benchmarks",
    projects_root: str = "projects",
) -> dict[str, Any]:
    raise NotImplementedError("populate_project will be implemented in Task 5")
