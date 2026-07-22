from __future__ import annotations

import logging
from typing import Any

from Bio import Entrez

from dargus.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ReportSearcher(BaseAgent):
    """Searches academic/web sources for literature and data."""

    name = "ReportSearcher"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.email = self.config.get("literature", {}).get("email", "dargus@example.com")

    def search(
        self,
        drug_ids: list[str],
        disease_id: str,
        max_results: int = 20,
    ) -> dict[str, list[Any]]:
        queries = self._build_queries(drug_ids, disease_id)
        downloaded: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []

        for query in queries:
            pmids = self._pubmed_search(query, max_results=max_results)
            if pmids:
                for pmid in pmids:
                    # MVP: do not download full text automatically; suggest manual access.
                    suggestions.append(
                        {
                            "source": "PubMed",
                            "pmid": pmid,
                            "query": query,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "reason": "Candidate literature",
                        }
                    )
            else:
                suggestions.append(
                    {
                        "source": "PubMed",
                        "query": query,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}",
                        "reason": "No automatic matches; manual search suggested",
                    }
                )

        # Always suggest key public databases
        suggestions.extend(self._database_suggestions(drug_ids, disease_id))

        return {"downloaded_files": downloaded, "manual_suggestions": suggestions}

    def _build_queries(self, drug_ids: list[str], disease_id: str) -> list[str]:
        queries = []
        for drug in drug_ids:
            queries.append(f"{drug} {disease_id}")
            queries.append(f"{drug} pharmacodynamics")
        return queries

    def _pubmed_search(self, query: str, max_results: int) -> list[str]:
        try:
            Entrez.email = self.email
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            return [str(pmid) for pmid in record.get("IdList", [])]
        except Exception as exc:
            logger.warning("PubMed search failed for %r: %s", query, exc)
            return []

    def _database_suggestions(self, drug_ids: list[str], disease_id: str) -> list[dict[str, Any]]:
        return [
            {
                "source": "ChEMBL",
                "url": "https://www.ebi.ac.uk/chembl/",
                "reason": "Compound bioactivity data",
            },
            {
                "source": "ClinicalTrials.gov",
                "url": f"https://clinicaltrials.gov/search?cond={disease_id.replace(' ', '+')}",
                "reason": "Clinical trial results",
            },
        ]

    def run(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute a literature/data search task.

        Returns dict for backward compat. For AgentReport use run_harness().
        """
        result = self.search(
            drug_ids=task_spec.get("drug_ids", []),
            disease_id=task_spec.get("disease_id", ""),
            max_results=task_spec.get("max_results", 20),
        )
        return {"status": "ok", **result}
