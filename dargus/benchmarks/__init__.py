"""Dargus benchmark package for training and evaluating Iris-* agents."""

from dargus.benchmarks.config import BENCHMARK_DATASETS, get_dataset_config
from dargus.benchmarks.datasets import BenchmarkDataset
from dargus.benchmarks.evaluator import IrisBenchmarkEvaluator
from dargus.benchmarks.reporter import BenchmarkReporter
from dargus.benchmarks.splitters import StratifiedDrugDiseaseEndpointSplitter
from dargus.benchmarks.trainer import IrisBenchmarkTrainer

__all__ = [
    "BENCHMARK_DATASETS",
    "get_dataset_config",
    "BenchmarkDataset",
    "StratifiedDrugDiseaseEndpointSplitter",
    "IrisBenchmarkTrainer",
    "IrisBenchmarkEvaluator",
    "BenchmarkReporter",
]
