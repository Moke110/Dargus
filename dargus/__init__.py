"""Dargus — clinical efficacy prediction system."""

__version__ = "0.15.4"

from dargus.api import (
    benchmark,
    ingest,
    predict,
    predict_single_agent,
    query_dbase,
    query_expert,
    status,
)
from dargus.dbase import DBase
from dargus.iris.commander import Iris


def __getattr__(name):
    if name == "train":
        import warnings

        warnings.warn(
            "'train' is deprecated, use 'ingest' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return ingest
    raise AttributeError(f"module 'dargus' has no attribute {name!r}")


__all__ = [
    "Iris",
    "DBase",
    "predict",
    "predict_single_agent",
    "query_dbase",
    "query_expert",
    "status",
    "benchmark",
    "ingest",
    "train",
    "__version__",
]
