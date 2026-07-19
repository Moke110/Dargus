"""Benchmark configuration registry."""

from __future__ import annotations

BENCHMARK_DATASETS: dict[str, dict] = {
    "top": {
        "name": "TOP",
        "description": (
            "Clinical trial outcome prediction benchmark from the TOP repository. "
            "Each row is a trial with drugs, diseases, and a binary approval label."
        ),
        "type": "drug_endpoint",
        "category": "top",
        "train": "data/toy_train.csv",
        "test": "data/toy_test.csv",
        "endpoint": "trial_success",
        "label_column": "label",
    },
    "hint": {
        "name": "HINT",
        "description": (
            "Heterogeneous network integration benchmark derived from the TOP repository. "
            "Same underlying trial data as TOP, used for graph-based models."
        ),
        "type": "drug_endpoint",
        "category": "top",
        "train": "data/toy_train.csv",
        "test": "data/toy_test.csv",
        "endpoint": "trial_success",
        "label_column": "label",
    },
}


def get_dataset_config(name: str) -> dict:
    if name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark dataset {name!r}")
    return BENCHMARK_DATASETS[name]
