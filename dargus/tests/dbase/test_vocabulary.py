"""Tests for VocabularyManager v0.15.0 — CURIE registry + enums."""

import tempfile
from pathlib import Path

from dargus.dbase.vocabulary import VocabularyManager


def test_vocab_enum_values():
    vm = VocabularyManager()
    assert "rct" in vm.get_enum("biological_level")
    assert "epi" in vm.get_enum("biological_level")
    assert "clinical" not in vm.get_enum("biological_level")
    assert len(vm.get_enum("biological_level")) == 11


def test_vocab_curie_validation():
    vm = VocabularyManager()
    assert vm.validate_curie("chembl:CHEMBL25")
    assert vm.validate_curie("uniprot:P35354")
    assert vm.validate_curie("mondo:0005180")
    assert not vm.validate_curie("chembl:bad_format")
    assert not vm.validate_curie("unknown:acc")


def test_vocab_save_load():
    vm = VocabularyManager()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        vm.save(path)
        vm2 = VocabularyManager.load(path)
        assert "rct" in vm2.get_enum("biological_level")
