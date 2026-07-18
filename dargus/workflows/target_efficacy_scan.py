"""Target-disease efficacy scan workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dargus.agents.base import new_task_id
from dargus.agents.director import DirectorAgent
from dargus.agents.epidemiology import EpiAgent
from dargus.agents.molecular import MoleculeAgent
from dargus.agents.retriever import RetrieverAgent
from dargus.agents.translate import TranslateAgent
from dargus.database import DataMaster
from dargus.reasoning import DirisEngine

logger = logging.getLogger(__name__)


def run(
    project_id: str | None = None,
    target: str | None = None,
    disease: str | None = None,
    clinical_endpoints: list[str] | None = None,
    drug_list: list[str] | None = None,
    director: DirectorAgent | None = None,
) -> dict[str, Any]:
    """Run the target efficacy scan workflow."""
    if director is None:
        director = DirectorAgent()

    if project_id is None:
        if disease is None:
            raise ValueError("Either project_id or disease must be provided")
        info = director.start_project(
            disease=disease,
            target=target,
            clinical_endpoints=clinical_endpoints,
        )
        project_id = info["project_id"]
        if clinical_endpoints is None:
            clinical_endpoints = director._default_endpoints(disease)
    else:
        project_dir = Path(director.projects_root) / project_id
        # Load config to get endpoints if not provided
        if clinical_endpoints is None:
            import yaml

            config_path = project_dir / "project_config.yaml"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
                    clinical_endpoints = cfg.get("clinical_endpoints", ["primary_endpoint_change"])
            else:
                clinical_endpoints = ["primary_endpoint_change"]

    config = director.config
    project_dir = Path(director.projects_root) / project_id

    # Step 1: Retrieve literature for molecular and epidemiology levels
    retriever = RetrieverAgent(config=config)
    mol_papers = retriever.search(
        query=f"{target} kinase inhibitor structure activity" if target else "drug efficacy",
        biological_level="molecular",
        max_papers=5,
    )
    epi_papers = retriever.search(
        query=f"{target} {disease} genetics GWAS" if target and disease else f"{disease} genetics",
        biological_level="epidemiology",
        max_papers=5,
    )

    # Step 2: Ingest a few synthetic samples for demonstration
    data_master = DataMaster(project_id, projects_root=str(director.projects_root))
    if drug_list is None:
        drug_list = ["LRRK2-IN-1", "DNL201", "MLi-2"] if target == "LRRK2" else ["Drug_A", "Drug_B"]
    samples = []
    for drug in drug_list:
        samples.append(
            {
                "sample_type": "in_vitro",
                "disease": disease or "unknown",
                "treatment": drug,
                "assay_endpoint": "IC50",
                "endpoint_category": "molecular",
                "source_type": "published_literature",
                "source_id": "PMID:00000000",
                "biological_level": "molecular",
                "data_type": "continuous",
                "data_label": "IC50",
                "data_value": 50.0,
                "data_unit": "nM",
                "data_quality": "medium",
            }
        )
    data_master.ingest(samples, source_type="published_literature", biological_level="molecular")

    # Step 3: Run level agents
    mol_task = {
        "project_id": project_id,
        "task_id": new_task_id(),
        "target_agent": "MoleculeAgent",
        "task_spec": {
            "target": target or "unknown",
            "disease": disease or "unknown",
            "molecules": ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
            "clinical_endpoints": clinical_endpoints,
            "task_name": "molecular_analysis",
        },
        "output_format": "standard_five_pack",
    }
    mol_agent = MoleculeAgent(config=config)
    mol_result = mol_agent.run(mol_task)

    epi_task = {
        "project_id": project_id,
        "task_id": new_task_id(),
        "target_agent": "EpiAgent",
        "task_spec": {
            "disease": disease or "unknown",
            "task_name": "epidemiology_analysis",
        },
        "output_format": "standard_five_pack",
    }
    epi_agent = EpiAgent(config=config)
    epi_result = epi_agent.run(epi_task)

    # Step 4: TranslateAgent
    trans_agent = TranslateAgent(config=config)
    trans_result = trans_agent.run(
        {
            "project_id": project_id,
            "task_spec": {"disease": disease or "unknown"},
        }
    )

    # Step 5: Diris
    level_embeddings = {}
    for drug in drug_list:
        level_embeddings[drug] = {
            "molecular": mol_result["level_embedding"],
            "epidemiology": epi_result["level_embedding"],
        }

    engine = DirisEngine(config=config)
    diris_result = engine.predict(
        project_id=project_id,
        drug_list=drug_list,
        clinical_endpoints=clinical_endpoints,
        level_embeddings=level_embeddings,
        translation_score=trans_result["translation_score"]["translation_score"],
    )

    return {
        "status": "ok",
        "project_id": project_id,
        "project_dir": str(project_dir),
        "literature": {"molecular": len(mol_papers), "epidemiology": len(epi_papers)},
        "agent_results": {
            "molecular": mol_result,
            "epidemiology": epi_result,
            "translation": trans_result,
        },
        "diris": diris_result,
    }


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
