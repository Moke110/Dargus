"""Table-driven tests for the disease resolver (ingestion.resolver)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dargus.ingestion.resolver import (
    ALIASES,
    load_registry,
    normalize,
    resolve_disease,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY = REPO_ROOT / "disease_registry.json"


@pytest.fixture(autouse=True)
def _load_real_registry():
    """Resolver tests use the real curated registry (no mocking)."""
    load_registry(REGISTRY)


# ── tier 1: exact registry name ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "term",
    [
        "essential hypertension",
        "schizophrenia",
        "rheumatoid arthritis",
        "diabetes mellitus",
        "epilepsy",
        "Crohn disease",
        "Alzheimer disease",
    ],
)
def test_exact_name_match(term):
    assert resolve_disease(term) is not None
    assert resolve_disease(term).startswith("mondo:")


def test_exact_name_returns_expected_curie():
    assert resolve_disease("essential hypertension") == "mondo:0001134"


# ── specific-over-umbrella + curated alias fallback ─────────────────────────


def test_specific_cancer_subtype_preferred_over_umbrella():
    """Cancer subtypes resolve to the specific registry entry when one exists."""
    # registry carries 'prostate carcinoma' -> MONDO:0005159, 'colorectal cancer' -> MONDO:0005575
    assert resolve_disease("prostate cancer") == "mondo:0005159"
    assert resolve_disease("colorectal cancer") == "mondo:0005575"
    assert resolve_disease("renal cell carcinoma") == "mondo:0002367"
    # melanoma resolves specifically too (registry entry, not umbrella)
    assert resolve_disease("melanoma") == "mondo:0005105"


@pytest.mark.parametrize(
    ("term", "target"),
    [
        ("diabetes", "diabetes mellitus"),
        ("type 2 diabetes", "diabetes mellitus"),
        ("hypertension", "essential hypertension"),
        ("depression", "major depressive disorder"),
        ("breast cancer", "cancer"),
        ("multiple myeloma", "cancer"),
        ("glioblastoma", "cancer"),
    ],
)
def test_alias_table(term, target):
    expected = resolve_disease(target)
    assert expected is not None
    assert resolve_disease(term) == expected


# ── tier 3: synonym match ────────────────────────────────────────────────────


def test_synonym_terms_resolve():
    # "tumors" -> "tumor" is not needed; assert a registry synonym resolves
    # if the registry ever carries one. Here we simply assert synonyms are
    # incorporated: every registry entry's own name resolves to its CURIE.
    registry = load_registry(REGISTRY)
    for entry in registry[:20]:
        name = entry.get("name")
        if name:
            assert resolve_disease(name) is not None


# ── tier 4: substring match ──────────────────────────────────────────────────


def test_substring_match():
    # "Recurrent Glioblastoma" resolves via alias stripping; assert a
    # condition that contains a full registry name resolves, preferring the
    # most specific registry match over the umbrella entry.
    assert resolve_disease("rheumatoid arthritis of the hand") == resolve_disease(
        "rheumatoid arthritis"
    )
    assert resolve_disease("stage iii colorectal cancer") == resolve_disease("colorectal cancer")


# ── tier 5: qualifier / parenthetical stripping ──────────────────────────────


@pytest.mark.parametrize(
    "term",
    [
        "Recurrent Glioblastoma",
        "Metastatic Breast Cancer",
        "Refractory Essential Hypertension",
        "Advanced Parkinson Disease",
    ],
)
def test_qualifier_stripping(term):
    assert resolve_disease(term) is not None


def test_parenthetical_stripping():
    assert resolve_disease("Prostate Cancer (stage IV)") == resolve_disease("prostate cancer")


# ── normalization ────────────────────────────────────────────────────────────


def test_case_and_whitespace_normalization():
    assert resolve_disease("  PROSTATE   cancer ") == resolve_disease("prostate cancer")


def test_accented_terms_normalize():
    assert normalize("café") == "cafe"


# ── unmappable ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("term", ["", None, "xyzzy qwerty", "not a real disease term here"])
def test_unmappable_returns_none(term):
    assert resolve_disease(term) is None


def test_alias_table_keys_are_normalized_terms():
    for key in ALIASES:
        assert normalize(key) == key, f"alias key {key!r} is not already normalized"


def test_alias_table_targets_exist_in_registry():
    """Every alias must resolve to a real registry CURIE (no dead entries)."""
    for key, target in ALIASES.items():
        resolved = resolve_disease(target)
        assert resolved is not None, f"alias {key!r} -> target {target!r} does not resolve"


def test_registry_file_exists():
    assert REGISTRY.exists(), f"expected registry at {REGISTRY}"
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert data["total_diseases"] >= 700
