from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VocabularyManager:
    """Maps ontology terms to integer factors per vocabulary namespace."""

    def __init__(self, vocabularies: dict[str, dict[str, int]] | None = None):
        self._vocab: dict[str, dict[str, int]] = vocabularies or {}
        self._synonyms: dict[str, dict[str, str]] = {}

    def get_or_create(self, vocab_name: str, term: str) -> int:
        """Return existing factor or assign a new integer."""
        vocab = self._vocab.setdefault(vocab_name, {})
        canonical = self._canonical(vocab_name, term)
        if canonical in vocab:
            return vocab[canonical]
        new_id = len(vocab)
        vocab[canonical] = new_id
        return new_id

    def get(self, vocab_name: str, term: str) -> int | None:
        vocab = self._vocab.get(vocab_name, {})
        canonical = self._canonical(vocab_name, term)
        return vocab.get(canonical)

    def add_synonym(self, vocab_name: str, synonym: str, canonical: str) -> None:
        self._synonyms.setdefault(vocab_name, {})[synonym] = canonical

    def _canonical(self, vocab_name: str, term: str) -> str:
        return self._synonyms.get(vocab_name, {}).get(term, term)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._vocab.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyManager":
        return cls(vocabularies=data.get("vocabularies", {}))

    def save(self, path: str | Path) -> None:
        payload = {
            "vocabularies": self.to_dict(),
            "synonyms": self._synonyms,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyManager":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vm = cls.from_dict(data)
        for vocab_name, synonyms in data.get("synonyms", {}).items():
            for syn, canonical in synonyms.items():
                vm.add_synonym(vocab_name, syn, canonical)
        return vm
