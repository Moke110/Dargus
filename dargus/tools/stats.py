"""Basic statistical utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float, float]:
    """Compute Cohen's d and its 95 % CI."""
    x = np.asarray(group1, dtype=float)
    y = np.asarray(group2, dtype=float)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    d = (np.mean(x) - np.mean(y)) / (pooled_std + 1e-12)
    se = np.sqrt(1 / nx + 1 / ny)
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se
    return float(d), float(ci_lower), float(ci_upper)


def compare_groups(values: np.ndarray, groups: np.ndarray, test: str = "auto") -> dict[str, Any]:
    """Run t-test or Mann-Whitney U based on normality heuristic."""
    x = values[groups == groups[0]]
    y = values[groups != groups[0]]
    if test == "auto":
        _, p_norm = stats.shapiro(values) if len(values) < 5000 else (0, 1)
        test = "t-test" if p_norm > 0.05 else "mann-whitney"
    if test == "t-test":
        stat, pvalue = stats.ttest_ind(x, y)
    else:
        stat, pvalue = stats.mannwhitneyu(x, y, alternative="two-sided")
    d, ci_lower, ci_upper = cohen_d(x, y)
    return {
        "test": test,
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "cohen_d": d,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }
