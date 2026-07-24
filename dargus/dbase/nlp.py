"""D-Base NLP v0.17.0 — embedding delegated to EmbeddingModel + evidence-to-text serialization."""

from __future__ import annotations

import numpy as np

from dargus.models.embedding import Embedding, EmbeddingModel


class MockNLP:
    """Fallback embedder returning zero vectors when sentence-transformers is unavailable."""

    def embed_text(self, text: str) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        return 0.0


class DBaseNLP:
    """Embedding backend for D-Base evidence records.

    Delegates embedding computation to an :class:`EmbeddingModel` instance.
    ``record_to_text`` (static) is the non-embedding serialization utility;
    it lives here unchanged.
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self._embedding_model = embedding_model

    # ── lazy default ────────────────────────────────────────────────────────

    def _get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            from dargus.models.embedding import SentenceTransformerBackend

            backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
            self._embedding_model = EmbeddingModel(backend)
        return self._embedding_model

    # ── embedding ──────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text, returning a numpy array (backward-compat)."""
        try:
            emb: Embedding = self._get_embedding_model().embed([text])[0]
            return np.array(emb, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return MockNLP().embed_text(text)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two texts."""
        try:
            model = self._get_embedding_model()
            ea: Embedding = model.embed([a])[0]
            eb: Embedding = model.embed([b])[0]
            return EmbeddingModel.similarity(ea, eb)
        except Exception:  # noqa: BLE001
            return MockNLP().similarity(a, b)

    @staticmethod
    def record_to_text(record: dict) -> str:
        """Serialize a v0.15.5 three-axis evidence dict to text for embedding.

        Incorporates llm_summary when present (§8.4).
        """
        parts: list[str] = []

        # llm_summary — primary semantic surface if present
        llm = record.get("llm_summary", "")
        if llm:
            parts.append(f"summary: {llm}")

        level = record.get("biological_level", "")
        if level:
            parts.append(f"level: {level}")

        design = record.get("evidence_design", "")
        if design:
            parts.append(f"design: {design}")

        # x axis (three-axis)
        x = record.get("x") or {}
        xtype = x.get("type", "")
        if xtype:
            parts.append(f"x.type: {xtype}")
        for item in x.get("value") or []:
            if isinstance(item, dict):
                eid = item.get("entity_id", "")
                elabel = item.get("entity_label", "")
                alt = item.get("alteration", "")
                parts.append(f"x: id={eid or elabel} alt={alt}")

        # y axis (three-axis)
        y = record.get("y") or {}
        yt = y.get("type", "")
        if yt:
            parts.append(f"y.type: {yt}")
        yc = y.get("category", "")
        if yc:
            parts.append(f"y.category: {yc}")
        yv = y.get("value") or []
        yu = y.get("unit", "")
        if yv:
            parts.append(f"y.value: {yv} {yu}".strip())
        ye = y.get("effect")
        if ye:
            parts.append(f"y.effect: {ye.get('type', '')}={ye.get('value')}")

        # bg axis
        bg = record.get("bg") or {}
        dids = bg.get("disease_id") or []
        if dids:
            parts.append(f"bg.disease_id: {dids}")

        # sample identity
        for k in ("cell_line_id", "model_organism", "strain", "sex", "tissue", "cell_type"):
            v = record.get(k)
            if v:
                parts.append(f"{k}: {v}")

        # clinical design
        cd = record.get("clinical_design") or {}
        for k, v in cd.items():
            if v:
                parts.append(f"{k}: {v}")

        # sources
        sources = record.get("sources", [])
        if sources:
            rank1 = next((s for s in sources if s.get("rank") == 1), None)
            if rank1:
                parts.append(f"source: {rank1.get('type')}:{rank1.get('id', '')}")

        # legacy fallback: interventions
        if not x.get("value") and "interventions" in record:
            for iv in record.get("interventions", []):
                label = iv.get("entity_label") or iv.get("entity_id", "unknown")
                role = iv.get("role", "")
                parts.append(f"{role} intervention: {label}")

        # legacy fallback: disease_id
        if not dids and record.get("disease_id"):
            parts.append(f"disease: {record.get('disease_id')}")

        # legacy fallback: readout
        if not yt and record.get("readout_type"):
            parts.append(f"readout: {record.get('readout_type')}")

        return "; ".join(parts)
