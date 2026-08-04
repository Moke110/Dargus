"""Disease resolver: free-text condition/indication strings to MONDO CURIEs.

Pure, I/O-free once the registry is loaded. Backed by the curated disease
registry (``disease_registry.json``) plus a small alias table that maps
observed conditions to registry entries. Used by the ingest converters so
clinical raw records (ClinicalTrials conditions, OpenFDA indications, ClinVar
traits) resolve to a registered ``mondo:`` CURIE or fail with an explicit,
logged skip reason.

Matching tiers, in order:
  1. exact name match against the registry
  2. curated alias table
  3. exact synonym match
  4. substring match (the condition contains a registry name)
  5. qualifier / parenthetical stripping ("Recurrent X" -> "X"), retry 1-4
  6. case/whitespace normalization is implicit in every tier; else ``None``
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

# ── registry loading (pure once loaded) ──────────────────────────────────────

_REGISTRY: list[dict] | None = None
_BY_NAME: dict[str, str] = {}
_BY_SYNONYM: dict[str, str] = {}


def _canon_mondo(mondo: str) -> str:
    """Normalize a registry CURIE to the validator's registered prefix case."""
    prefix, _, accession = mondo.partition(":")
    return f"{prefix.lower()}:{accession}"


def load_registry(path: str | Path | None = None) -> list[dict]:
    """Load the curated disease registry (list of {name, mondo_id, ...})."""
    global _REGISTRY, _BY_NAME, _BY_SYNONYM
    if _REGISTRY is not None and path is None:
        return _REGISTRY
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "disease_registry.json"
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    diseases = data["diseases"] if isinstance(data, dict) else data
    _REGISTRY = diseases
    _BY_NAME = {}
    _BY_SYNONYM = {}
    for d in diseases:
        name = d.get("name")
        mondo = d.get("mondo_id")
        if not name or not mondo:
            continue
        curie = _canon_mondo(mondo)
        _BY_NAME[normalize(name)] = curie
        for syn in _synonyms(d):
            _BY_SYNONYM.setdefault(normalize(syn), curie)
    return _REGISTRY


def _synonyms(d: dict) -> Iterable[str]:
    for key in ("synonyms", "aliases"):
        for s in d.get(key) or []:
            if isinstance(s, str):
                yield s


# ── normalization ─────────────────────────────────────────────────────────────

_STOPWORDS = {"of", "the", "and", "with", "without", "due", "to", "in", "for", "a", "an"}


def normalize(term: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", term)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in s.split() if t not in _STOPWORDS}


# ── qualifier stripping ───────────────────────────────────────────────────────

_RECURRENCE = re.compile(r"^(recurrent|recurring|relapsed|metastatic|advanced|refractory)\b")
_PARENTHETICAL = re.compile(r"\(.*?\)")


def _strip_qualifiers(term: str) -> str:
    s = normalize(term)
    prev = None
    while s != prev:
        prev = s
        s = _RECURRENCE.sub("", s).strip()
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_parenthetical(term: str) -> str:
    return _PARENTHETICAL.sub("", term).strip()


# ── curated alias table (observed conditions → canonical registry name) ──────
#
# Values are registry ``name`` values (not CURIEs) so the lookup stays
# resilient to registry edits. Only broad, defensible umbrella mappings are
# listed: cancer-site subtypes -> the registry's umbrella ``cancer`` entry,
# diabetes variants -> ``diabetes mellitus``, etc. No invented CURIEs.

ALIASES: dict[str, str] = {
    # oncology — umbrella ``cancer`` (MONDO:0004992) for common subtypes
    "prostate cancer": "cancer",
    "breast cancer": "cancer",
    "non small cell lung cancer": "cancer",
    "small cell lung cancer": "cancer",
    "lung cancer": "cancer",
    "ovarian cancer": "cancer",
    "colorectal cancer": "cancer",
    "pancreatic cancer": "cancer",
    "renal cell carcinoma": "cancer",
    "hepatocellular carcinoma": "cancer",
    "gastric cancer": "cancer",
    "bladder cancer": "cancer",
    "head and neck cancer": "cancer",
    "melanoma": "cancer",
    "multiple myeloma": "cancer",
    "leukemia": "cancer",
    "lymphoma": "cancer",
    "glioblastoma": "cancer",
    "solid tumor": "cancer",
    "hematologic malignancy": "cancer",
    "metastatic cancer": "cancer",
    "advanced cancer": "cancer",
    "hereditary cancer predisposing syndrome": "cancer",
    # metabolic / endocrine
    "diabetes": "diabetes mellitus",
    "type 2 diabetes": "diabetes mellitus",
    "type 1 diabetes": "diabetes mellitus",
    "type 2 diabetes mellitus": "diabetes mellitus",
    "type 1 diabetes mellitus": "diabetes mellitus",
    # cardiovascular
    "hypertension": "essential hypertension",
    "blood pressure high": "essential hypertension",
    "heart disease": "heart failure",
    "congestive heart failure": "heart failure",
    "stroke": "cerebrovascular disorder",
    "coronary artery disease": "cerebrovascular disorder",
    "cerebrovascular disease": "cerebrovascular disorder",
    # respiratory / infection
    "hiv": "hiv infectious disease",
    "hiv infection": "hiv infectious disease",
    "hiv infections": "hiv infectious disease",
    "tuberculosis": "tuberculosis",
    # neuro / psychiatric
    "depression": "major depressive disorder",
    "major depressive disorder": "major depressive disorder",
    "migraine": "migraine with aura",
    "headache": "migraine with aura",
    "obesity": "obesity disorder",
}


# ── main resolver ─────────────────────────────────────────────────────────────


def resolve_disease(term: str) -> str | None:
    """Return a ``mondo:`` CURIE for *term*, or ``None`` if unmappable."""
    if not term or not isinstance(term, str):
        return None
    load_registry()

    s = normalize(term)
    if not s:
        return None

    # tier 1 — exact registry name
    if s in _BY_NAME:
        return _BY_NAME[s]

    # tier 2 — curated alias
    if s in ALIASES:
        target = normalize(ALIASES[s])
        if target in _BY_NAME:
            return _BY_NAME[target]

    # tier 3 — exact synonym
    if s in _BY_SYNONYM:
        return _BY_SYNONYM[s]

    # tier 4 — substring (condition contains a registry name); among all
    # registry names fully contained in the condition, prefer the most
    # specific (longest) one so "stage iii colorectal cancer" resolves to
    # colorectal cancer, not the umbrella cancer entry.
    tokens = _tokens(s)
    best: tuple[int, str] | None = None
    for name, mondo in _BY_NAME.items():
        name_tokens = _tokens(name)
        if name_tokens and name_tokens <= tokens:
            if best is None or len(name_tokens) > best[0]:
                best = (len(name_tokens), mondo)
    if best is not None:
        return best[1]

    # tier 5 — strip qualifiers / parentheticals, retry
    stripped = _strip_qualifiers(_strip_parenthetical(term))
    if stripped and stripped != s:
        return resolve_disease(stripped)

    return None
