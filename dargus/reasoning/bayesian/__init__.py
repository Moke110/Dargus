"""Bayesian modeling for Dargus Iris-bayes (pymc-dependent, lazy-loaded)."""

__all__ = ["HierarchicalBayesianModel"]


def __getattr__(name):
    if name == "HierarchicalBayesianModel":
        from dargus.reasoning.bayesian.model import HierarchicalBayesianModel as _M

        return _M
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
