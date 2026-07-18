"""Ontology normalization stubs.

In Phase 0 these return identity mappings. Future phases will integrate
MONDO, HGNC, DrugBank/PubChem, and OBI lookup services.
"""

from __future__ import annotations


def normalize_disease(name: str) -> dict[str, str]:
    """Return a standardized disease record."""
    return {"input": name, "standard_name": name, "ontology_id": ""}


def normalize_drug(name: str) -> dict[str, str]:
    """Return a standardized drug record."""
    return {"input": name, "standard_name": name, "drugbank_id": "", "pubchem_cid": ""}


def normalize_gene(symbol: str) -> dict[str, str]:
    """Return a standardized gene record."""
    return {"input": symbol, "symbol": symbol, "hgnc_id": ""}


def normalize_assay(assay_name: str) -> dict[str, str]:
    """Return a standardized assay record."""
    return {"input": assay_name, "obi_term": assay_name, "obi_id": ""}
