from __future__ import annotations

import json
import logging
from typing import Any

from dargus.iris.base import IrisAgent, PredictionMatrix, normalize_prediction_entry
from dargus.models.compat import DargusLLM, llm_from_config

logger = logging.getLogger(__name__)


class IrisLlm(IrisAgent):
    """LLM-based prediction from a text summary of D-Base."""

    name = "Iris-llm"

    def __init__(
        self,
        backend: DargusLLM | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.backend = backend or llm_from_config(config)

    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        summary, supporting_map = self._summarize_dbase(dbase, drug_ids, disease_id)
        prompt = self._build_prompt(summary, drug_ids, disease_id, endpoints)
        raw = self.backend.complete(prompt)
        parsed = self._parse_output(raw, drug_ids, endpoints)

        for drug in drug_ids:
            drug_entries = parsed.setdefault(drug, {})
            for endpoint in endpoints:
                entry = drug_entries.setdefault(endpoint, {})
                entry.setdefault("normalized_effect_size", 0.0)
                entry.setdefault("ci95_lower", -0.5)
                entry.setdefault("ci95_upper", 0.5)
                entry.setdefault("supporting_records", supporting_map.get(drug, []))
                entry.setdefault("reasoning_mode", self.name)
                entry.setdefault("confidence_level", "exploratory")
                drug_entries[endpoint] = normalize_prediction_entry(
                    entry,
                    reasoning_mode=self.name,
                    confidence_level="exploratory",
                )
        return parsed

    def _summarize_dbase(
        self, dbase: Any, drug_ids: list[str], disease_id: str
    ) -> tuple[str, dict[str, list[str]]]:
        lines = [f"Disease: {disease_id}", f"Drugs: {', '.join(drug_ids)}", "Records:"]
        supporting: dict[str, list[str]] = {drug: [] for drug in drug_ids}
        for drug in drug_ids:
            records = dbase.query(drug_id=drug, disease_id=disease_id)
            for r in records:
                supporting[drug].append(r.record_id)
                note = r.provenance_note
                if note:
                    lines.append(f"- {r.record_id}: {note}")
                else:
                    lines.append(f"- {r.record_id}")
        return "\n".join(lines), supporting

    def _build_prompt(
        self, summary: str, drug_ids: list[str], disease_id: str, endpoints: list[str]
    ) -> str:
        return f"""You are a clinical pharmacology assistant.
Given the following evidence summary, predict the normalized effect size
(Cohen's d scale divided by MCID) for each drug on each clinical endpoint.
Return ONLY a JSON object with this exact structure:
{{
  "drug_id": {{
    "endpoint_name": {{
      "normalized_effect_size": 0.0,
      "ci95_lower": 0.0,
      "ci95_upper": 0.0,
      "reasoning": "short explanation"
    }}
  }}
}}

Summary:
{summary}

Drugs: {drug_ids}
Disease: {disease_id}
Endpoints: {endpoints}
"""

    def _parse_output(
        self, raw: str, drug_ids: list[str], endpoints: list[str]
    ) -> PredictionMatrix:
        try:
            # Extract JSON if wrapped in markdown
            text = raw.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as exc:
            logger.warning("Failed to parse Iris-llm output: %s", exc)
            return {
                drug: {
                    endpoint: {
                        "normalized_effect_size": 0.0,
                        "ci95_lower": -0.5,
                        "ci95_upper": 0.5,
                    }
                    for endpoint in endpoints
                }
                for drug in drug_ids
            }
