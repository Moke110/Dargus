"""Bayesian hierarchical model for Diris Phase 0."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def predict_bayesian(
    drug_list: list[str],
    drug_embeddings: dict[str, np.ndarray],
    disease_embedding: np.ndarray,
    level_embeddings: dict[str, dict[str, np.ndarray]],
    translation_score: dict[str, Any],
    clinical_endpoints: list[str],
    n_samples: int = 2000,
    n_chains: int = 4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Return predictions with 95 % credible intervals.

    If PyMC is available, run a small hierarchical model. Otherwise fall back to
    an analytic prior-only prediction.
    """
    try:

        return _pymc_predict(
            drug_list,
            drug_embeddings,
            disease_embedding,
            level_embeddings,
            translation_score,
            clinical_endpoints,
            n_samples,
            n_chains,
            random_seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMC unavailable or failed (%s); using analytic fallback", exc)
        return _analytic_predict(
            drug_list,
            drug_embeddings,
            disease_embedding,
            level_embeddings,
            translation_score,
            clinical_endpoints,
            random_seed,
        )


def _collect_layer_signals(
    drug: str,
    level_embeddings: dict[str, dict[str, np.ndarray]],
    translation_score: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect available layer embeddings and their weights."""
    layer_order = ["molecular", "cellular", "exvivo", "animal", "clinical", "epidemiology"]
    signals = []
    weights = []
    layer_specific = translation_score.get("translation_score", {}).get("layer_specific", {})
    overall = translation_score.get("translation_score", {}).get("overall", 0.5)
    for layer in layer_order:
        emb = level_embeddings.get(drug, {}).get(layer)
        if emb is None:
            continue
        # Map layer to translation weight
        key_map = {
            "molecular": "molecular_to_clinical",
            "cellular": "cellular_to_animal",
            "exvivo": "exvivo_to_animal",
            "animal": "animal_to_clinical",
            "clinical": "animal_to_clinical",
            "epidemiology": "animal_to_clinical",
        }
        w = layer_specific.get(key_map.get(layer, ""), overall)
        # Reduce embedding to scalar signal (mean of effect dimensions)
        signal = float(np.mean(emb[:192])) if len(emb) >= 192 else float(np.mean(emb))
        signals.append(signal)
        weights.append(float(w))
    if not signals:
        return np.array([0.0]), np.array([overall])
    return np.array(signals, dtype=float), np.array(weights, dtype=float)


def _pymc_predict(
    drug_list: list[str],
    drug_embeddings: dict[str, np.ndarray],
    disease_embedding: np.ndarray,
    level_embeddings: dict[str, dict[str, np.ndarray]],
    translation_score: dict[str, Any],
    clinical_endpoints: list[str],
    n_samples: int,
    n_chains: int,
    random_seed: int,
) -> dict[str, Any]:
    import pymc as pm

    rng = np.random.default_rng(random_seed)
    predictions: dict[str, Any] = {}

    for endpoint in clinical_endpoints:
        predictions[endpoint] = {}
        for drug in drug_list:
            signals, weights = _collect_layer_signals(drug, level_embeddings, translation_score)
            drug_emb = drug_embeddings.get(drug, rng.normal(size=256))
            prior_mean = float(np.mean(drug_emb)) * 0.1 + float(np.mean(disease_embedding)) * 0.1

            with pm.Model():
                # Hyper-prior for clinical effect
                mu = pm.Normal("mu", mu=prior_mean, sigma=1.0)
                tau = pm.HalfNormal("tau", sigma=1.0)

                # Layer observations
                layer_effects = []
                for i, (signal, weight) in enumerate(zip(signals, weights)):
                    scaled = signal * weight
                    pm.Normal(f"layer_{i}", mu=mu + scaled, sigma=tau, observed=scaled)
                    layer_effects.append(scaled)

                # Predictive
                pm.Deterministic("effect", mu)

                try:
                    trace = pm.sample(
                        n_samples // n_chains,
                        chains=n_chains,
                        random_seed=random_seed,
                        progressbar=False,
                        cores=1,
                    )
                    effect_samples = trace.posterior["effect"].values.flatten()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "PyMC sampling failed for %s/%s (%s); fallback",
                        drug,
                        endpoint,
                        exc,
                    )
                    effect_samples = rng.normal(loc=prior_mean, scale=0.5, size=n_samples)

                mean = float(np.mean(effect_samples))
                ci_lower, ci_upper = float(np.percentile(effect_samples, 2.5)), float(
                    np.percentile(effect_samples, 97.5)
                )
                prob_placebo = float(np.mean(np.array(effect_samples) > 0))
                prob_mcid = float(np.mean(np.array(effect_samples) > 0.5))

                predictions[endpoint][drug] = {
                    "normalized_effect_size": round(mean, 3),
                    "ci_95_lower": round(ci_lower, 3),
                    "ci_95_upper": round(ci_upper, 3),
                    "probability_superior_to": {
                        "placebo": round(prob_placebo, 3),
                        "clinically_meaningful": round(prob_mcid, 3),
                    },
                }

    return {"model": "bayesian_pymc", "predictions": predictions}


def _analytic_predict(
    drug_list: list[str],
    drug_embeddings: dict[str, np.ndarray],
    disease_embedding: np.ndarray,
    level_embeddings: dict[str, dict[str, np.ndarray]],
    translation_score: dict[str, Any],
    clinical_endpoints: list[str],
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    predictions: dict[str, Any] = {}
    overall = translation_score.get("translation_score", {}).get("overall", 0.5)

    for endpoint in clinical_endpoints:
        predictions[endpoint] = {}
        for drug in drug_list:
            signals, weights = _collect_layer_signals(drug, level_embeddings, translation_score)
            drug_emb = drug_embeddings.get(drug, rng.normal(size=256))
            prior_mean = float(np.mean(drug_emb)) * 0.1 + float(np.mean(disease_embedding)) * 0.1
            weighted = np.sum(signals * weights) / (np.sum(weights) + 1e-9)
            mean = prior_mean * (1 - overall) + weighted * overall
            # Uncertainty scales inversely with evidence weight and sample count
            uncertainty = 1.0 / (1.0 + np.sum(weights))
            ci_lower = float(mean - 1.96 * uncertainty)
            ci_upper = float(mean + 1.96 * uncertainty)
            predictions[endpoint][drug] = {
                "normalized_effect_size": round(mean, 3),
                "ci_95_lower": round(ci_lower, 3),
                "ci_95_upper": round(ci_upper, 3),
                "probability_superior_to": {
                    "placebo": round(float(mean > 0), 3),
                    "clinically_meaningful": round(float(mean > 0.5), 3),
                },
            }

    return {"model": "bayesian_analytic_fallback", "predictions": predictions}
