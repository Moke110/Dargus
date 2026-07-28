"""D-Base vocabulary manager — three-axis enum registry loaded from vocabularies.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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
        entry = self._data.get("y_effect_value_type") or {}
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
