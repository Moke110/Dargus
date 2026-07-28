"""D-Base vocabulary manager — three-axis enum registry loaded from vocabularies.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── module-level compat exports (loaded from vocabularies.json) ──────────────

_vm: VocabularyManager | None = None


def _get_vm() -> "VocabularyManager":
    global _vm
    if _vm is None:
        _vm = VocabularyManager()
    return _vm


# backward-compatible module-level enum lists (used by existing tests)
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
    """CURIE prefix registry + three-axis enum term registry.

    Loads all controlled vocabularies from vocabularies.json (§3.0–§3.14).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._curie_patterns: dict[str, re.Pattern] = {}
        self._load()

    def _load(self) -> None:
        path = Path(__file__).resolve().parent / "vocabularies.json"
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

        # compile CURIE patterns
        curie_data = self._data.get("curie_prefixes") or {}
        for prefix, pat in (curie_data.get("hard_validated") or {}).items():
            self._curie_patterns[prefix] = re.compile(pat)

    def get_enum_values(self, vocab_name: str) -> list[str]:
        """Return flat list of enum values for a vocabulary."""
        entry = self._data.get(vocab_name) or {}
        if isinstance(entry, dict):
            vals = entry.get("values", [])
            if vals and isinstance(vals[0], dict):
                return [item["value"] for item in vals]
            return list(vals)
        return []

    def get_enum(self, vocab_name: str) -> list[str]:
        """Alias for get_enum_values."""
        return self.get_enum_values(vocab_name)

    def is_valid_enum_value(self, vocab_name: str, value: str) -> bool:
        return value in self.get_enum_values(vocab_name)

    def clinical_levels(self) -> frozenset:
        entry = self._data.get("biological_level") or {}
        vals = entry.get("values", [])
        return frozenset(item["value"] for item in vals if item.get("is_clinical"))

    def sim_levels(self) -> frozenset:
        entry = self._data.get("biological_level") or {}
        vals = entry.get("values", [])
        return frozenset(item["value"] for item in vals if item.get("is_sim"))

    def log_effect_types(self) -> frozenset:
        entry = self._data.get("y_effect_type") or {}
        return frozenset(entry.get("log_types", []))

    def control_labels(self) -> frozenset:
        return frozenset(self.get_enum_values("x_value_control_labels"))

    def validate_curie(self, curie_str: str) -> bool:
        """Check if a CURIE string has a registered prefix and valid accession."""
        if ":" not in curie_str:
            return False
        prefix, _, accession = curie_str.partition(":")

        curie_data = self._data.get("curie_prefixes") or {}
        fallback = set(curie_data.get("fallback", []))

        if prefix in fallback:
            return True
        pat = self._curie_patterns.get(prefix)
        if pat is None:
            return False
        return bool(pat.match(accession))

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyManager":
        vm = cls()
        if Path(path).exists():
            vm._data = json.loads(Path(path).read_text(encoding="utf-8"))
        return vm
