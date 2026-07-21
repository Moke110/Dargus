"""Benchmark runner: copy blank → backfill → train → evaluate → discard."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from dargus.benchmarks.evaluator import IrisBenchmarkEvaluator
from dargus.benchmarks.extractor import BenchmarkExtractor
from dargus.benchmarks.reporter import BenchmarkReporter
from dargus.benchmarks.splitters import StratifiedDrugDiseaseEndpointSplitter
from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.dbase.paths import default_dargus_home
from dargus.iris.commander import Iris

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrate benchmark conditions using DBase-blank and DBase-bench."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.work_dir = Path(default_dargus_home() / "benchmark")
        self.bench_dir = self.work_dir / "bench"
        self.extractor = BenchmarkExtractor(work_dir=self.work_dir)

    def run(self) -> dict[str, Any]:
        strip = self.config.get("strip", {})
        records, blank = self.extractor.extract(strip)
        if not records:
            raise ValueError("Strip condition matched no records.")

        conditions = self.config.get("conditions", [])
        all_metrics: dict[str, dict[str, dict[str, float]]] = {}
        all_predictions: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

        for condition in conditions:
            name = condition["name"]
            train_df, test_df = self._split(records, condition.get("split", {}))
            if train_df.empty:
                logger.warning("Condition %s has empty train split; skipping.", name)
                continue
            self._prepare_bench(blank)
            self._backfill(train_df)
            predictions = self._evaluate(test_df)
            metrics = IrisBenchmarkEvaluator().evaluate(predictions, test_df)
            all_metrics[name] = metrics
            all_predictions[name] = predictions
            self._discard_bench()

        reporter = BenchmarkReporter(output_dir=self.work_dir / "outputs")
        paths = reporter.report(self._flatten_metrics(all_metrics), all_predictions)
        shutil.rmtree(self.extractor.blank_dir, ignore_errors=True)
        return {
            "metrics": str(paths["metrics"]),
            "predictions": str(paths["predictions"]),
            "conditions": list(all_metrics.keys()),
        }

    def _split(self, records: list, split_cfg: dict[str, Any]):
        rows = []
        for r in records:
            manager = DBaseManager(DBase.global_instance())
            fc = manager._record_field(r, "fold_change")
            if fc is None:
                continue
            rows.append(
                {
                    "drug_id": manager._record_field(r, "drug_id"),
                    "disease_id": manager._record_field(r, "disease_id"),
                    "endpoint": manager._record_field(r, "endpoint"),
                    "label": 1 if float(fc) > 0 else 0,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df, df
        test_size = split_cfg.get("test_size", 0.2)
        random_state = split_cfg.get("random_state", 42)
        return StratifiedDrugDiseaseEndpointSplitter(
            test_size=test_size, random_state=random_state
        ).split(df)

    def _prepare_bench(self, blank: DBase) -> None:
        if self.bench_dir.exists():
            shutil.rmtree(self.bench_dir)
        self.bench_dir.mkdir(parents=True)
        shutil.copytree(blank.dbase_dir, self.bench_dir / "dbase")

    def _backfill(self, train_df) -> None:
        dbase = DBase("global", root_dir=self.bench_dir)
        manager = DBaseManager(dbase)
        for _, row in train_df.iterrows():
            record = manager.fill_template(
                {
                    "biological_level": "clinical",
                    "drug_id": row["drug_id"],
                    "disease_id": row["disease_id"],
                    "endpoint": row["endpoint"],
                    "fold_change": 0.5 if row["label"] == 1 else -0.5,
                },
                source_metadata={"type": "benchmark_backfill"},
                suggested_template="clinical_trial_outcome_v1",
            )
            manager.write_record(record, dedup=False)

    def _evaluate(self, test_df) -> dict[str, dict[str, dict[str, Any]]]:
        iris = Iris()
        drug_ids = test_df["drug_id"].unique().tolist()
        diseases = test_df["disease_id"].unique().tolist()
        endpoints = test_df["endpoint"].unique().tolist()
        predictions: dict[str, dict[str, dict[str, Any]]] = {}
        for disease_id in diseases:
            preds = iris.infer(
                drug_ids=drug_ids,
                disease_id=disease_id,
                endpoints=endpoints,
                confirm_callback=lambda plan: True,
            )
            for drug, eps in preds.items():
                if drug not in predictions:
                    predictions[drug] = {}
                predictions[drug].update(eps)
        return {"Iris": predictions}

    def _discard_bench(self) -> None:
        if self.bench_dir.exists():
            shutil.rmtree(self.bench_dir, ignore_errors=True)

    def _flatten_metrics(
        self, all_metrics: dict[str, dict[str, dict[str, float]]]
    ) -> dict[str, dict[str, float]]:
        flattened: dict[str, dict[str, float]] = {}
        for condition, agents in all_metrics.items():
            for agent, metrics in agents.items():
                key = f"{condition}.{agent}"
                flattened[key] = metrics
        return flattened
