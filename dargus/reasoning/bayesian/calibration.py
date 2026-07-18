from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dargus.reasoning.bayesian.model import HierarchicalBayesianModel


def calibrate_priors(
    y: np.ndarray,
    level_idx: np.ndarray,
    group_idx: np.ndarray,
    output_path: str | Path,
    draws: int = 300,
    tune: int = 300,
) -> dict[str, Any]:
    model = HierarchicalBayesianModel(y, level_idx, group_idx)
    model.fit(draws=draws, tune=tune)

    mu_global = float(model.trace.posterior["mu_global"].mean())
    sigma_global = float(model.trace.posterior["sigma_global"].mean())
    level_bias = model.trace.posterior["level_bias"].mean(axis=(0, 1)).values.tolist()
    level_sigma = model.trace.posterior["level_sigma"].mean(axis=(0, 1)).values.tolist()

    priors = {
        "mu_global": mu_global,
        "sigma_global": sigma_global,
        "level_bias": level_bias,
        "level_sigma": level_sigma,
    }
    Path(output_path).write_text(json.dumps(priors, indent=2), encoding="utf-8")
    return priors
