"""Benchmark evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def _point_prediction(low: float, up: float) -> float:
    return (low + up) / 2.0


class IrisBenchmarkEvaluator:
    """Compute classification and interval-quality metrics for predictions."""

    def evaluate(
        self,
        predictions: dict[str, dict[str, dict[str, dict[str, Any]]]],
        test_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Evaluate each agent's predictions against the test labels."""
        if test_df.empty:
            raise ValueError("test_df must not be empty")
        if not predictions:
            raise ValueError("predictions must not be empty")

        metrics: dict[str, dict[str, float]] = {}
        for agent_name, agent_preds in predictions.items():
            metrics[agent_name] = self._evaluate_agent(agent_preds, test_df)
        return metrics

    def _evaluate_agent(
        self,
        agent_preds: dict[str, dict[str, dict[str, Any]]],
        test_df: pd.DataFrame,
    ) -> dict[str, float]:
        y_true: list[int] = []
        y_prob: list[float] = []
        widths: list[float] = []
        covered: list[bool] = []

        for _, row in test_df.iterrows():
            drug = row["drug_id"]
            endpoint = row["endpoint"]
            label = int(row["label"])
            pred = agent_preds.get(drug, {}).get(endpoint)
            if pred is None:
                raise ValueError(f"Missing prediction for ({drug}, {endpoint})")
            low = float(pred["efficacy_low"])
            up = float(pred["efficacy_up"])
            prob = _point_prediction(low, up)
            y_true.append(label)
            y_prob.append(prob)
            widths.append(up - low)
            covered.append(low <= label <= up)

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        if len(set(y_true)) > 1:
            auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
            auprc = float(average_precision_score(y_true_arr, y_prob_arr))
        else:
            auroc = 0.0
            auprc = 0.0
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))
        ece = self._expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10)
        coverage = float(np.mean(covered))
        width = float(np.mean(widths))
        uncertainty_error_corr = self._uncertainty_error_correlation(
            y_true_arr, y_prob_arr, np.array(widths)
        )

        return {
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "brier": brier,
            "ece": ece,
            "coverage": coverage,
            "width": width,
            "uncertainty_error_corr": uncertainty_error_corr,
            "n": len(y_true),
        }

    def _expected_calibration_error(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if i == 0:
                in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if not np.any(in_bin):
                continue
            avg_confidence = float(np.mean(y_prob[in_bin]))
            avg_accuracy = float(np.mean(y_true[in_bin]))
            ece += np.sum(in_bin) * abs(avg_confidence - avg_accuracy)
        return float(ece / len(y_true)) if len(y_true) > 0 else 0.0

    def _uncertainty_error_correlation(
        self, y_true: np.ndarray, y_prob: np.ndarray, widths: np.ndarray
    ) -> float:
        errors = np.abs(y_true - y_prob)
        if len(errors) < 2 or np.std(widths) == 0 or np.std(errors) == 0:
            return 0.0
        return float(np.corrcoef(widths, errors)[0, 1])
