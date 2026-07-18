"""Diris full-stack reasoning engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dargus.embedding import get_embedding_provider
from dargus.reasoning.models.bayesian import predict_bayesian

logger = logging.getLogger(__name__)


class DirisEngine:
    """Predicts normalized clinical effect sizes and confidence intervals."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def predict(
        self,
        project_id: str,
        drug_list: list[str],
        clinical_endpoints: list[str],
        level_embeddings: dict[str, dict[str, Any]] | None = None,
        translation_score: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full-stack prediction."""
        drug_emb_provider = get_embedding_provider(self.config, "drug")
        disease_emb_provider = get_embedding_provider(self.config, "disease")

        drug_embeddings = {drug: drug_emb_provider.encode(drug) for drug in drug_list}
        # Disease embedding uses the first endpoint name as a placeholder;
        # future versions will encode the project disease from config.
        disease_embedding = disease_emb_provider.encode(clinical_endpoints[0])

        # Convert level embedding dicts to numpy arrays
        level_arrays: dict[str, dict[str, np.ndarray]] = {}
        if level_embeddings:
            for drug, layers in level_embeddings.items():
                level_arrays[drug] = {}
                for layer, emb in layers.items():
                    if isinstance(emb, dict):
                        vec = np.array(emb.get("vector", []), dtype=float)
                    else:
                        vec = np.array(emb, dtype=float)
                    level_arrays[drug][layer] = vec

        result = predict_bayesian(
            drug_list=drug_list,
            drug_embeddings=drug_embeddings,
            disease_embedding=disease_embedding,
            level_embeddings=level_arrays,
            translation_score=(
                {"translation_score": translation_score}
                if translation_score
                else {"translation_score": {"overall": 0.5, "layer_specific": {}}}
            ),
            clinical_endpoints=clinical_endpoints,
            n_samples=self.config.get("diris", {}).get("n_samples", 2000),
            n_chains=self.config.get("diris", {}).get("n_chains", 4),
            random_seed=self.config.get("diris", {}).get("random_seed", 42),
        )

        # Write synthesis outputs
        project_dir = Path(self.config.get("projects", {}).get("root_dir", "projects")) / project_id
        synthesis_dir = project_dir / "synthesis"
        synthesis_dir.mkdir(parents=True, exist_ok=True)

        engine_input = {
            "project_id": project_id,
            "drug_list": drug_list,
            "clinical_endpoints": clinical_endpoints,
            "level_embeddings": {k: list(v.keys()) for k, v in (level_embeddings or {}).items()},
            "translation_score": translation_score,
        }
        (synthesis_dir / "engine_input.json").write_text(
            json.dumps(engine_input, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (synthesis_dir / "predictions.json").write_text(
            json.dumps(result["predictions"], indent=2, ensure_ascii=False), encoding="utf-8"
        )

        report = self._synthesis_report(result["predictions"], clinical_endpoints, drug_list)
        (synthesis_dir / "synthesis_report.md").write_text(report, encoding="utf-8")

        return {"status": "ok", "result": result, "synthesis_dir": str(synthesis_dir)}

    def _synthesis_report(
        self, predictions: dict[str, Any], endpoints: list[str], drugs: list[str]
    ) -> str:
        lines = ["# Dargus Synthesis Report", ""]
        lines.append("## Predicted normalized effect sizes (Cohen's d / MCID scale)")
        lines.append("")
        lines.append("| Drug | Endpoint | Effect size | 95 % CI | P > placebo | P > MCID |")
        lines.append("|------|----------|-------------|---------|---------------|-------------|")
        for endpoint in endpoints:
            for drug in drugs:
                pred = predictions[endpoint][drug]
                lines.append(
                    f"| {drug} | {endpoint} | {pred['normalized_effect_size']} | "
                    f"[{pred['ci_95_lower']}, {pred['ci_95_upper']}] | "
                    f"{pred['probability_superior_to']['placebo']} | "
                    f"{pred['probability_superior_to']['clinically_meaningful']} |"
                )
        lines.append("")
        lines.append("## Interpretation")
        lines.append("- Effect size ≈ 0: no predicted clinical benefit.")
        lines.append("- Effect size ≈ 1: predicted to reach MCID.")
        lines.append("- Wide CI: sparse or conflicting evidence.")
        lines.append("")
        lines.append("## Evidence gaps")
        lines.append("- Cell, ex vivo, animal, and clinical layers not populated in Phase 0 MVP.")
        lines.append("")
        lines.append(
            "> **Disclaimer**: Dargus outputs are for research purposes only "
            "and do not constitute clinical advice."
        )
        return "\n".join(lines)
