"""RetrieverAgent — unified literature retrieval."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from dargus.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """Search literature and extract structured information."""

    name = "RetrieverAgent"

    def search(
        self,
        query: str,
        biological_level: str,
        max_papers: int = 50,
        include_full_text: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a structured list of papers."""
        self._trace(
            project_id=self._project_id_from_task({}),
            task_id="n/a",
            event="tool_call",
            details={"query": query, "level": biological_level},
        )
        try:
            from Bio import Entrez

            email = self.config.get("retriever", {}).get("email") or "dargus@example.com"
            Entrez.email = email
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_papers)
            record = Entrez.read(handle)
            handle.close()
            id_list = record.get("IdList", [])
            results = []
            if id_list:
                summary_handle = Entrez.esummary(db="pubmed", id=",".join(id_list))
                summaries = Entrez.read(summary_handle)
                summary_handle.close()
                for paper in summaries:
                    title = paper.get("Title", "")
                    abstract = paper.get("Source", "")  # fallback
                    relevance = self._keyword_relevance(title + " " + abstract, query)
                    results.append(
                        {
                            "pmid": paper.get("Id"),
                            "title": title,
                            "authors": [a.get("Name", "") for a in paper.get("AuthorList", [])],
                            "year": (
                                paper.get("PubDate", "").split()[0] if paper.get("PubDate") else ""
                            ),
                            "abstract": abstract,
                            "relevance": relevance,
                            "biological_level": biological_level,
                            "doi": paper.get("DOI", ""),
                        }
                    )
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:max_papers]
        except Exception as exc:  # noqa: BLE001
            logger.warning("PubMed search failed (%s); returning offline fallback", exc)
            return self._offline_fallback(query, biological_level, max_papers)

    def extract(self, pdf_path: str) -> dict[str, Any]:
        """Extract text from a PDF."""
        path = Path(pdf_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {pdf_path}"}
        try:
            import fitz

            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return {"status": "ok", "text": text[:100000]}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc)}

    def _keyword_relevance(self, text: str, query: str) -> float:
        text = text.lower()
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            return 0.0
        return sum(1 for t in terms if t in text) / len(terms)

    def _offline_fallback(
        self, query: str, biological_level: str, max_papers: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "pmid": f"OFFLINE{i}",
                "title": f"Offline placeholder paper for {query}",
                "authors": ["Dargus Placeholder"],
                "year": "2026",
                "abstract": "This is an offline fallback result used when PubMed is unavailable.",
                "relevance": 1.0,
                "biological_level": biological_level,
                "doi": "",
            }
            for i in range(min(max_papers, 3))
        ]

    def _project_id_from_task(self, task_spec: dict[str, Any]) -> str:
        return task_spec.get("project_id", "default")

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute a retrieval task."""
        result = self.search(
            query=task_spec["task_spec"]["query"],
            biological_level=task_spec["task_spec"].get("biological_level", "molecular"),
            max_papers=task_spec["task_spec"].get("max_papers", 50),
        )
        return {"status": "ok", "results": result}
