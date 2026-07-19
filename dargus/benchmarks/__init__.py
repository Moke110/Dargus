"""Dargus benchmark package for training and evaluating Iris-* agents."""

from dargus.benchmarks.config import BENCHMARK_DATASETS, get_dataset_config
from dargus.benchmarks.datasets import BenchmarkDataset
from dargus.benchmarks.splitters import StratifiedDrugDiseaseEndpointSplitter

__all__ = [
    "BENCHMARK_DATASETS",
    "get_dataset_config",
    "BenchmarkDataset",
    "StratifiedDrugDiseaseEndpointSplitter",
]
