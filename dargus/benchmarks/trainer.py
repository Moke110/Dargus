"""Benchmark trainer for Iris-* agents."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dargus.iris.base import IrisAgent, PredictionMatrix


class IrisBenchmarkTrainer:
    """Train and predict with a list of Iris-* agents on a benchmark split."""

    def __init__(self, agents: list[IrisAgent]):
        self.agents = agents

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dbase: Any,
        embeddings: dict[str, Any] | None = None,
    ) -> dict[str, PredictionMatrix]:
        """Fit agents on the training split and predict on the test split."""
        predictions: dict[str, PredictionMatrix] = {}
        for agent in self.agents:
            if hasattr(agent, "fit"):
                agent.fit(train_df, dbase=dbase, embeddings=embeddings)
            agent_predictions = self._predict_agent(agent, test_df, dbase, embeddings)
            predictions[agent.name] = agent_predictions
        return predictions

    def _predict_agent(
        self,
        agent: IrisAgent,
        test_df: pd.DataFrame,
        dbase: Any,
        embeddings: dict[str, Any] | None,
    ) -> PredictionMatrix:
        drug_ids = test_df["drug_id"].unique().tolist()
        diseases = test_df["disease_id"].unique().tolist()
        endpoints = test_df["endpoint"].unique().tolist()
        # Predict per disease to respect IrisAgent.predict signature.
        result: PredictionMatrix = {drug: {} for drug in drug_ids}
        for disease_id in diseases:
            pred = agent.predict(
                dbase,
                drug_ids=drug_ids,
                disease_id=disease_id,
                endpoints=endpoints,
                embeddings=embeddings,
            )
            for drug in pred:
                result[drug].update(pred[drug])
        return result
