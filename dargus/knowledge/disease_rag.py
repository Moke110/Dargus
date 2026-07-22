"""DiseaseRAG — disease-domain knowledge retrieval (stub for v0.16.0)."""

from __future__ import annotations

import logging

from dargus.knowledge.base import KnowledgeItem, KnowledgeRetriever

logger = logging.getLogger(__name__)


class DiseaseRAG(KnowledgeRetriever):
    """Disease ontology and gene-disease association knowledge retrieval.

    In v0.16.0, this is a stub that returns empty results with graceful
    degradation. The full vector index will be populated during 0.17.x Ingest.
    """

    def __init__(self):
        self._index: dict[str, list[KnowledgeItem]] = {}  # entity_type -> items

    def search(
        self,
        query: str,
        domain: str | None = None,
        biological_level: str | None = None,
        top_k: int = 10,
    ) -> list[KnowledgeItem]:
        logger.debug("DiseaseRAG.search: query=%r (stub)", query[:80])
        return []

    def lookup(self, entity_id: str, entity_type: str) -> KnowledgeItem | None:
        logger.debug("DiseaseRAG.lookup: %s/%s (stub)", entity_type, entity_id)
        return None

    def index_items(self, items: list[KnowledgeItem]) -> None:
        """Index knowledge items for future retrieval (populated in 0.17.x)."""
        for item in items:
            self._index.setdefault(item.entity_type, []).append(item)
        logger.info(
            "DiseaseRAG: indexed %d items across %d types",
            len(items),
            len(self._index),
        )
