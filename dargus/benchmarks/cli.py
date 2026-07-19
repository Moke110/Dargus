"""Benchmark CLI orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dargus.benchmarks.datasets import BenchmarkDataset
from dargus.benchmarks.evaluator import IrisBenchmarkEvaluator
from dargus.benchmarks.reporter import BenchmarkReporter
from dargus.benchmarks.splitters import StratifiedDrugDiseaseEndpointSplitter
from dargus.benchmarks.trainer import IrisBenchmarkTrainer
from dargus.dbase import DBase, TemplateSchema
from dargus.dbase.manager import DBaseManager
from dargus.iris.base import IrisAgent
from dargus.iris.commander import Iris


class BenchmarkCli:
    """Orchestrate a full benchmark run: data → split → D-Base → train → predict → report."""

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "data",
        projects_root: str = "projects/benchmark",
        output_dir: str = "outputs/benchmark",
        agents: list[IrisAgent] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.projects_root = Path(projects_root)
        self.output_dir = Path(output_dir)
        self.agents = agents or []
        self.test_size = test_size
        self.random_state = random_state
        self.project_id = f"benchmark_{dataset_name}"

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the dataset and split into train/test."""
        dataset = BenchmarkDataset(self.dataset_name, data_root=str(self.data_root))
        df = dataset.load(split="train")
        splitter = StratifiedDrugDiseaseEndpointSplitter(
            test_size=self.test_size, random_state=self.random_state
        )
        return splitter.split(df)

    def populate_dbase(self, train_df: pd.DataFrame) -> DBase:
        """Write training records into a project D-Base."""
        project_dir = self.projects_root / self.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        templates_dir = project_dir / "dbase" / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        schema = TemplateSchema(
            template_id="benchmark_trial_v1",
            fields=[
                {"name": "biological_level", "type": "factor", "vocabulary": ["clinical"]},
                {"name": "drug_id", "type": "factor", "vocabulary_ref": "global_drug_vocab"},
                {"name": "disease_id", "type": "factor", "vocabulary_ref": "global_disease_vocab"},
                {"name": "endpoint", "type": "factor", "vocabulary_ref": "global_endpoint_vocab"},
                {"name": "fold_change", "type": "float"},
                {"name": "ci95_lower", "type": "float"},
                {"name": "ci95_upper", "type": "float"},
            ],
        )
        schema.to_yaml(templates_dir / "benchmark_trial_v1.yaml")
        dbase = DBase(self.project_id, root_dir=str(self.projects_root))
        dbase.add_template(schema)
        manager = DBaseManager(dbase)

        for _, row in train_df.iterrows():
            # Encode label as a synthetic fold_change: positive label -> +0.5, negative -> -0.5
            effect = 0.5 if int(row["label"]) == 1 else -0.5
            record = manager.fill_template(
                {
                    "biological_level": "clinical",
                    "drug_id": row["drug_id"],
                    "disease_id": row["disease_id"],
                    "endpoint": row["endpoint"],
                    "fold_change": effect,
                    "ci95_lower": effect - 0.5,
                    "ci95_upper": effect + 0.5,
                },
                source_metadata={"type": "benchmark", "dataset": self.dataset_name},
                suggested_template="benchmark_trial_v1",
            )
            manager.write_record(record)
        dbase.save()
        return dbase

    def run(self) -> dict[str, Any]:
        """Run the full benchmark pipeline."""
        train_df, test_df = self.load_and_split()
        dbase = self.populate_dbase(train_df)

        # Use provided agents or default to DiseaseExpert via Iris commander.
        agents = self.agents
        if not agents:
            iris = Iris(config={"projects": {"root_dir": str(self.projects_root)}})
            # DiseaseExpert is the default agent in Iris.predict context.
            predictions = iris.predict(
                project_id=self.project_id,
                drug_ids=test_df["drug_id"].unique().tolist(),
                disease_id=test_df["disease_id"].iloc[0],
                endpoints=test_df["endpoint"].unique().tolist(),
                confirm_callback=self._noop_confirm,
            )
            predictions = {"Iris": predictions}
        else:
            trainer = IrisBenchmarkTrainer(agents=agents)
            predictions = trainer.run(train_df, test_df, dbase=dbase)

        evaluator = IrisBenchmarkEvaluator()
        metrics = evaluator.evaluate(predictions, test_df)

        reporter = BenchmarkReporter(output_dir=str(self.output_dir))
        paths = reporter.report(metrics, predictions)

        return {
            "project_id": self.project_id,
            "metrics": paths["metrics"],
            "predictions": paths["predictions"],
            "output_dir": str(self.output_dir),
            "n_train": len(train_df),
            "n_test": len(test_df),
        }

    @staticmethod
    def _noop_confirm(_plan: dict[str, Any]) -> bool:
        return True
