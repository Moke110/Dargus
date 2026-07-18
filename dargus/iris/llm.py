from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from dargus.iris.base import IrisAgent, PredictionMatrix

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockLLMBackend:
    """Default backend for tests and offline runs."""

    def complete(self, prompt: str) -> str:
        return json.dumps({})


class IrisLlm(IrisAgent):
    """LLM-based prediction from a text summary of D-Base."""

    name = "Iris-llm"

    def __init__(self, backend: LLMBackend | None = None):
        self.backend = backend or MockLLMBackend()

    def predict(
        self,
        dbase: Any,
        drug_ids: list[str],
        disease_id: str,
        endpoints: list[str],
        embeddings: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionMatrix:
        summary = self._summarize_dbase(dbase, drug_ids, disease_id)
        prompt = self._build_prompt(summary, drug_ids, disease_id, endpoints)
        raw = self.backend.complete(prompt)
        parsed = self._parse_output(raw, drug_ids, endpoints)

        for drug in drug_ids:
            for endpoint in endpoints:
                entry = parsed.get(drug, {}).get(endpoint, {})
                if entry:
                    entry.setdefault("supporting_records", [])
                    entry.setdefault("reasoning_mode", self.name)
                    entry.setdefault("confidence_level", "exploratory")
        return parsed

    def _summarize_dbase(self, dbase: Any, drug_ids: list[str], disease_id: str) -> str:
        lines = [f"Disease: {disease_id}", f"Drugs: {', '.join(drug_ids)}", "Records:"]
        for drug in drug_ids:
            records = dbase.query(drug_id=drug, disease_id=disease_id)
            for r in records:
                lines.append(f"- {r.record_id}: {r.provenance_note}")
        return "\n".join(lines)

    def _build_prompt(
        self, summary: str, drug_ids: list[str], disease_id: str, endpoints: list[str]
    ) -> str:
        return f"""You are a clinical pharmacology assistant.
Given the following evidence summary, predict the normalized effect size (Cohen's d scale divided by MCID) for each drug on each clinical endpoint.
Return ONLY a JSON object with this exact structure:
{{
  "drug_id": {{
    "endpoint_name": {{
      "normalized_effect_size": float,
      "ci95_lower": float,
      "ci95_upper": float,
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

    def _parse_output(self, raw: str, drug_ids: list[str], endpoints: list[str]) -> PredictionMatrix:
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
            return {drug: {} for drug in drug_ids}
