"""EpiAgent — epidemiology-level analysis stub."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dargus.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class EpiAgent(BaseAgent):
    """Produces an epidemiology-level five-pack, even on empty data."""

    name = "EpiAgent"

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        project_id = task_spec["project_id"]
        task_id = task_spec.get("task_id", "unknown")
        spec = task_spec.get("task_spec", {})
        disease = spec.get("disease", "unknown")

        self._trace(
            project_id=project_id,
            task_id=task_id,
            event="started",
            details={"disease": disease},
        )

        dim = self.config.get("level_embedding", {}).get("dimension", 512)
        vector = np.zeros(dim, dtype=float)
        vector[-1] = 1.0  # missing-data marker

        embedding = {
            "layer": "epidemiology",
            "embedding_version": "1.0",
            "dimension": dim,
            "vector": vector.tolist(),
            "interpretable_summary": {
                "top_findings": ["No epidemiology data ingested for this project."],
                "evidence_quality": {"score": 0.0, "breakdown": {}},
                "key_uncertainties": ["GWAS/MR/rare-variant evidence not available"],
                "n_samples_analyzed": 0,
                "n_tools_used": 0,
            },
        }

        report = f"""# Epidemiology Analysis Report: {disease}

## Summary
No epidemiology samples were available for {disease} in this project.

## Methods
- Placeholder stub (Phase 0)

## Results
- Samples analyzed: 0
- Level embedding: missing-data marker set

## Evidence Quality Assessment
- Data quality: low (no data)

## Key Output Files
- report.md
- level_embedding.json

## Methodological Limitations & Caveats
This Phase 0 stub cannot assess GWAS, MR, or rare-variant evidence.

## Open Questions
- Are GWAS summary statistics available?
- Are credible instrument variables known for Mendelian randomization?

> **Disclaimer**: Dargus outputs are for research purposes only and do not
> constitute clinical advice.
"""

        paths = self.write_five_pack(
            project_id=project_id,
            layer="epidemiology",
            task_name=spec.get("task_name", "epidemiology_analysis"),
            report=report,
            figures=None,
            data=None,
            code=None,
            embedding=embedding,
        )

        self._trace(
            project_id=project_id,
            task_id=task_id,
            event="completed",
            details={"outputs": paths},
        )

        return {"status": "ok", "outputs": paths, "level_embedding": embedding}
