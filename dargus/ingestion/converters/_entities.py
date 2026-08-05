"""Shared entity resolution for the ingest converters.

Pure lookups over the curated drug list (``fda_approved_drugs.json``) so that
raw drug strings (ClinicalTrials interventions, OpenFDA label drugs) resolve
to a registered ``chembl:`` CURIE or fall back to ``entity_label`` only —
never an invented CURIE (per ticket #72's conservatism rule).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DRUGS: list[dict] | None = None
_BY_NAME: dict[str, dict] = {}
_BY_BRAND: dict[str, dict] = {}
_BY_CID: dict[str, dict] = {}


def load_drugs(path: str | Path | None = None) -> list[dict]:
    """Load the curated drug list (list of {name, chembl_id, ...})."""
    global _DRUGS, _BY_NAME, _BY_BRAND, _BY_CID
    if _DRUGS is not None and path is None:
        return _DRUGS
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "ingest"
            / "retrieval_v1"
            / "lists"
            / "fda_approved_drugs.json"
        )
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    drugs = data["drugs"] if isinstance(data, dict) else data
    _DRUGS = drugs
    _BY_NAME = {}
    _BY_BRAND = {}
    _BY_CID = {}
    for d in drugs:
        name = d.get("name")
        if not name:
            continue
        key = name.lower().strip()
        _BY_NAME.setdefault(key, d)
        for b in d.get("brands") or []:
            _BY_BRAND.setdefault(str(b).lower().strip(), d)
        if d.get("pubchem_cid"):
            _BY_CID[str(d["pubchem_cid"]).lower().strip()] = d
    return _DRUGS


def canon_mondo(curie: str) -> str:
    """Normalize a CURIE to the validator's registered prefix case."""
    prefix, _, accession = curie.partition(":")
    return f"{prefix.lower()}:{accession}"


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", term.lower().strip())


def resolve_drug(term: str) -> tuple[str | None, str]:
    """Resolve a drug string to (chembl_curie | None, canonical_label).

    Conservative: an exact match against the curated drug list (name, brand,
    or PubChem CID) yields a ``chembl:`` CURIE; otherwise the entity is
    emitted as ``entity_label`` with ``entity_id=None``. Never invents a CURIE.
    """
    if not term or not isinstance(term, str):
        return None, ""
    load_drugs()
    key = _norm(term)
    if not key:
        return None, ""
    d = _BY_NAME.get(key) or _BY_BRAND.get(key) or _BY_CID.get(key)
    if d:
        curie = f"chembl:{d['chembl_id']}" if d.get("chembl_id") else None
        return curie, d["name"]
    return None, term.strip()
