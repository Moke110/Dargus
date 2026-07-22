"""KnowledgeRetriever — unified interface for all knowledge sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeItem:
    entity_id: str
    entity_type: str  # "disease" | "drug" | "target"
    content: str  # LLM-injectable text
    source: str  # "disease_rag" | "dbase" | "chembl"
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeRetriever(ABC):
    """Unified interface for all knowledge sources."""

    @abstractmethod
    def search(
        self,
        query: str,
        domain: str | None = None,
        biological_level: str | None = None,
        top_k: int = 10,
    ) -> list[KnowledgeItem]: ...

    @abstractmethod
    def lookup(self, entity_id: str, entity_type: str) -> KnowledgeItem | None: ...
