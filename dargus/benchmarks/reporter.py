"""Benchmark reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class BenchmarkReporter:
    """Write benchmark metrics and predictions to CSV files."""

    def __init__(self, output_dir: str | Path = "outputs/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report(
        self,
        metrics: dict[str, dict[str, float]],
        predictions: dict[str, dict[str, dict[str, dict[str, Any]]]],
    ) -> dict[str, Path]:
        """Write metrics.csv and predictions.csv to the output directory."""
        metrics_path = self.output_dir / "metrics.csv"
        predictions_path = self.output_dir / "predictions.csv"

        metrics_rows = []
        for agent, agent_metrics in metrics.items():
            row = {"agent": agent}
            row.update(agent_metrics)
            metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

        prediction_rows = []
        for agent, agent_preds in predictions.items():
            for drug, endpoints in agent_preds.items():
                for endpoint, pred in endpoints.items():
                    prediction_rows.append(
                        {
                            "agent": agent,
                            "drug_id": drug,
                            "endpoint": endpoint,
                            "efficacy_low": pred.get("efficacy_low"),
                            "efficacy_up": pred.get("efficacy_up"),
                            "reasoning_mode": pred.get("reasoning_mode"),
                            "confidence_level": pred.get("confidence_level"),
                        }
                    )
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

        return {"metrics": metrics_path, "predictions": predictions_path}
