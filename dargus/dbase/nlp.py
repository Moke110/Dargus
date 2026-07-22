"""D-Base NLP v0.15.0 — embedding backend + evidence-to-text serialization."""

from __future__ import annotations

import numpy as np


class MockNLP:
    """Fallback embedder returning zero vectors when sentence-transformers is unavailable."""

    def embed_text(self, text: str) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        return 0.0


class DBaseNLP:
    """Embedding backend for D-Base evidence records and DiseaseRAG knowledge bases."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:  # noqa: BLE001
                return MockNLP().embed_text(text)
        try:
            return self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        except Exception:  # noqa: BLE001
            return MockNLP().embed_text(text)

    def similarity(self, a: str, b: str) -> float:
        va = self.embed_text(a)
        vb = self.embed_text(b)
        norm = float(np.linalg.norm(va)) * float(np.linalg.norm(vb))
        if norm == 0:
            return 0.0
        return float(np.dot(va, vb) / norm)

    @staticmethod
    def record_to_text(record: dict) -> str:
        """Serialize an evidence dict to natural-language text for embedding.

        No reverse_lookup needed — evidence is already human-readable.
        """
        parts: list[str] = []

        # Interventions
        for iv in record.get("interventions", []):
            label = iv.get("entity_label") or iv.get("entity_id", "unknown")
            role = iv.get("role", "")
            parts.append(f"{role} intervention: {label}")

        # Disease
        disease = record.get("disease_id", "")
        if disease:
            parts.append(f"disease: {disease}")

        # Level
        level = record.get("biological_level", "")
        if level:
            parts.append(f"level: {level}")

        # Readout
        rt = record.get("readout_type", "")
        if rt:
            parts.append(f"readout: {rt}")
        rv = record.get("readout_value")
        ru = record.get("readout_unit", "")
        if rv is not None:
            parts.append(f"value: {rv} {ru}".strip())

        # Effect
        effect = record.get("effect") or {}
        if effect:
            etype = effect.get("type", "")
            evalue = effect.get("value")
            if etype and evalue is not None:
                parts.append(f"effect: {etype}={evalue}")

        # Context
        ec = record.get("experimental_context") or {}
        if ec:
            for k in ("model_method", "cell_line_id", "model_organism", "strain"):
                v = ec.get(k)
                if v:
                    parts.append(f"{k}: {v}")

        cd = record.get("clinical_design") or {}
        if cd:
            for k in ("phase", "population", "study_id"):
                v = cd.get(k)
                if v:
                    parts.append(f"{k}: {v}")

        # Sources summary
        sources = record.get("sources", [])
        if sources:
            rank1 = next((s for s in sources if s.get("rank") == 1), None)
            if rank1:
                parts.append(f"source: {rank1.get('type')}:{rank1.get('id', '')}")

        return "; ".join(parts)
