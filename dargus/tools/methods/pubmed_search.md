---
tool: pubmed_search
level: any
assay_type: literature_retrieval
database_source: "PubMed (NCBI)"
key_columns: [pmid, title, abstract, publication_date, authors]
limitations: "Rate-limited by NCBI E-utilities API; max 3 requests/second without API key"
---
# PubMed Search

## Method
E-utilities-based PubMed search via Biopython Entrez.
Constructs MeSH-tagged queries from drug/disease context.

## Output Interpretation
- `pmid`: PubMed unique identifier
- `abstract`: truncated at 2000 characters for LLM context windows
- Results ranked by relevance (Best Match algorithm)
