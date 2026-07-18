from __future__ import annotations

import numpy as np
import pymc as pm

LEVEL_ORDER = ["molecular", "cellular", "exvivo", "animal", "clinical"]


class HierarchicalBayesianModel:
    def __init__(
        self,
        y: np.ndarray,
        level_idx: np.ndarray,
        group_idx: np.ndarray,
        n_levels: int = 5,
    ):
        self.y = np.asarray(y, dtype=float)
        self.level_idx = np.asarray(level_idx, dtype=int)
        self.group_idx = np.asarray(group_idx, dtype=int)
        self.n_levels = n_levels
        self.n_groups = int(self.group_idx.max() + 1) if self.group_idx.size else 1
        self.trace = None

    def fit(self, draws: int = 500, tune: int = 500, chains: int = 2) -> None:
        with pm.Model() as self.model:
            mu_global = pm.Normal("mu_global", 0, 1)
            sigma_global = pm.HalfNormal("sigma_global", 1)

            level_bias = pm.Normal("level_bias", 0, 1, shape=self.n_levels)
            level_sigma = pm.HalfNormal("level_sigma", 1, shape=self.n_levels)

            group_effect = pm.Normal(
                "group_effect",
                mu=0,
                sigma=sigma_global,
                shape=self.n_groups,
            )

            mu = mu_global + level_bias[self.level_idx] + group_effect[self.group_idx]
            sigma = level_sigma[self.level_idx]

            pm.Normal("obs", mu=mu, sigma=sigma, observed=self.y)

            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                progressbar=False,
                random_seed=42,
            )

    def predict(self, level: int, group: int) -> dict:
        if self.trace is None:
            raise RuntimeError("Model has not been fitted")

        mu_global = self.trace.posterior["mu_global"].values.flatten()
        level_bias = self.trace.posterior["level_bias"].values.reshape(-1, self.n_levels)
        group_effect = self.trace.posterior["group_effect"].values.reshape(-1, self.n_groups)

        samples = mu_global + level_bias[:, level] + group_effect[:, group]
        mean = float(np.mean(samples))
        ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
        return {
            "mean": mean,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
