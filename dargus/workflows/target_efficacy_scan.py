"""Target-disease efficacy scan workflow."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run(**kwargs: Any) -> dict[str, Any]:
    """Deprecated v3 workflow. Use run_v4() instead."""
    raise NotImplementedError(
        "The v3 target_efficacy_scan workflow has been removed. Use run_v4() instead."
    )


def run_v4(
    drugs: list[str],
    disease: str,
    datadir: str | None = None,
    projects_root: str = "projects",
) -> dict:
    """Run the v4.0 target-disease efficacy scan workflow."""
    from dargus.agents.director import DirectorAgent
    from dargus.dbase import DBase
    from dargus.iris.selector import IrisSelector

    director = DirectorAgent(config={"projects": {"root_dir": projects_root}})
    project = director.start_project(disease=disease)
    director.run_workflow_v4(
        "target_efficacy_scan",
        project["project_id"],
        drug_ids=drugs,
        disease_id=disease,
        datadir=datadir,
    )
    dbase = DBase(project["project_id"], root_dir=projects_root)
    selector = IrisSelector(dbase)
    predictions = selector.predict(drugs, disease)
    return {"project_id": project["project_id"], "predictions": predictions}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a target-disease efficacy scan")
    parser.add_argument("--target", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--endpoints", nargs="+")
    parser.add_argument("--drugs", nargs="+")
    args = parser.parse_args()
    result = run(
        target=args.target,
        disease=args.disease,
        clinical_endpoints=args.endpoints,
        drug_list=args.drugs,
    )
    print(f"Workflow completed: {result['project_id']}")
    print(f"Project directory: {result['project_dir']}")
    print(f"Predictions: {result['diris']['result']['predictions']}")
