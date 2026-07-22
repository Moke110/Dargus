from unittest.mock import patch

from dargus.agents.report_searcher import ReportSearcher


def test_search_returns_suggestions_when_empty():
    agent = ReportSearcher()
    with patch.object(agent, "_pubmed_search", return_value=[]):
        result = agent.search(["LRRK2-IN-1"], "Parkinson's disease")
    assert result["downloaded_files"] == []
    assert any("PubMed" in s["source"] for s in result["manual_suggestions"])


def test_search_returns_per_pmid_suggestions():
    agent = ReportSearcher()
    with patch.object(agent, "_pubmed_search", side_effect=[["12345"], ["67890"]]):
        result = agent.search(["LRRK2-IN-1"], "Parkinson's disease")

    pubmed = [s for s in result["manual_suggestions"] if s["source"] == "PubMed"]
    assert len(pubmed) == 2
    pmids = {s["pmid"] for s in pubmed}
    assert pmids == {"12345", "67890"}
    assert all(s["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/") for s in pubmed)


def test_search_includes_database_suggestions():
    agent = ReportSearcher()
    with patch.object(agent, "_pubmed_search", return_value=[]):
        result = agent.search(["LRRK2-IN-1"], "Parkinson's disease")

    sources = {s["source"] for s in result["manual_suggestions"]}
    assert "ChEMBL" in sources
    assert "ClinicalTrials.gov" in sources

    clinical = next(s for s in result["manual_suggestions"] if s["source"] == "ClinicalTrials.gov")
    assert "Parkinson%27s+disease" in clinical["url"] or "Parkinson" in clinical["url"]


def test_build_queries():
    agent = ReportSearcher()
    queries = agent._build_queries(["Drug-A", "Drug-B"], "Some disease")
    assert queries == [
        "Drug-A Some disease",
        "Drug-A pharmacodynamics",
        "Drug-B Some disease",
        "Drug-B pharmacodynamics",
    ]


def test_pubmed_search_failure_returns_empty():
    agent = ReportSearcher()
    with patch("dargus.agents.report_searcher.Entrez.esearch", side_effect=RuntimeError("network")):
        assert agent._pubmed_search("query", 10) == []


def test_run_delegates_to_search():
    agent = ReportSearcher()
    with patch.object(agent, "_pubmed_search", return_value=["111"]):
        result = agent.run(
            {
                "drug_ids": ["X"],
                "disease_id": "Y",
                "max_results": 5,
            }
        )
    assert result["status"] == "ok"
    assert result["downloaded_files"] == []
    assert any(s.get("pmid") == "111" for s in result["manual_suggestions"])
