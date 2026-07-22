"""D-Base vocabulary manager v0.15.0 — CURIE prefix registry + closed enum values."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── closed enum vocabularies (from field_registry.yaml) ──────────────────────

BIOLOGICAL_LEVELS = [
    "molecular",
    "molecular-sim",
    "cellular",
    "cellular-sim",
    "exvivo",
    "exvivo-sim",
    "animal",
    "animal-sim",
    "rct",
    "epi",
    "rct-sim",
]

READOUT_CATEGORIES = [
    "clinic_efficacy_primary",
    "clinic_efficacy_secondary",
    "clinic_efficacy_exploratory",
    "clinic_toxicity_primary",
    "clinic_toxicity_secondary",
    "clinic_toxicity_exploratory",
    "binding",
    "pk_adme",
    "prot_exp",
    "rna_exp",
    "viability",
    "apoptosis",
    "proliferation",
    "migration",
    "invasion",
    "autophagy",
    "differentiation",
    "phosphorylation",
    "localization",
    "metabolism",
    "oxidative_stress",
    "behavioral",
    "other",
]

EVIDENCE_DESIGNS = [
    "two_arm_comparison",
    "single_arm",
    "dose_escalation",
    "dose_response_curve",
    "observational_association",
    "continuous_trajectory",
    "descriptive",
]

# CURIE prefix registry
CURIE_PATTERNS: dict[str, str] = {
    "chembl": r"^CHEMBL\d+$",
    "uniprot": r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$",
    "mondo": r"^\d{7}$",
    "doid": r"^\d+$",
    "hp": r"^\d{7}$",
    "meddra": r"^\d{8}$",
    "uberon": r"^\d{7}$",
    "cl": r"^\d{7}$",
    "cellosaurus": r"^CVCL_[A-Z0-9]{4}$",
    "NCBITaxon": r"^\d+$",
    "clinicaltrials": r"^NCT\d{8}$",
}

FALLBACK_PREFIXES = frozenset(
    {
        "drugbank",
        "rxnorm",
        "unii",
        "iuphar",
        "pubchem.compound",
        "complexportal",
        "refseq",
        "genbank",
        "insdc",
        "bto",
    }
)


class VocabularyManager:
    """CURIE prefix registry + enum term registry (v0.15.0)."""

    def __init__(self) -> None:
        self._enums: dict[str, list[str]] = {
            "biological_level": BIOLOGICAL_LEVELS,
            "readout_category": READOUT_CATEGORIES,
            "evidence_design": EVIDENCE_DESIGNS,
        }
        self._curie_patterns: dict[str, re.Pattern] = {}
        for prefix, pattern in CURIE_PATTERNS.items():
            self._curie_patterns[prefix] = re.compile(pattern)

    def validate_curie(self, curie_str: str) -> bool:
        """Check if a CURIE string has a registered prefix and valid accession."""
        if ":" not in curie_str:
            return False
        prefix, _, accession = curie_str.partition(":")
        if prefix in FALLBACK_PREFIXES:
            return True
        pat = self._curie_patterns.get(prefix)
        if pat is None:
            return False
        return bool(pat.match(accession))

    def get_enum(self, vocab_name: str) -> list[str]:
        """Return the values for a closed enum vocabulary."""
        return list(self._enums.get(vocab_name, []))

    def is_valid_enum_value(self, vocab_name: str, value: str) -> bool:
        return value in self._enums.get(vocab_name, [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "biological_level": BIOLOGICAL_LEVELS,
            "readout_category": READOUT_CATEGORIES,
            "evidence_design": EVIDENCE_DESIGNS,
            "curie_prefixes": {k: p for k, p in CURIE_PATTERNS.items()},
            "fallback_prefixes": sorted(FALLBACK_PREFIXES),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyManager":
        vm = cls()
        if Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for k in ("biological_level", "readout_category", "evidence_design"):
                if k in data:
                    vm._enums[k] = data[k]
            if "curie_prefixes" in data:
                for prefix, pattern in data["curie_prefixes"].items():
                    vm._curie_patterns[prefix] = re.compile(pattern)
        return vm
