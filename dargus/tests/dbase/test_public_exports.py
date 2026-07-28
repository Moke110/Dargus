"""Tests for public exports."""

from dargus.dbase import DBase, DBaseStore, VocabularyManager
from dargus.experts import BiomedExpert, Expert, MoleculeExpert


def test_dbase_public_exports():
    assert callable(DBase)
    assert callable(DBaseStore)


def test_vocabulary_public_exports():
    assert callable(VocabularyManager)


def test_expert_public_exports():
    assert issubclass(MoleculeExpert, Expert)
    assert issubclass(BiomedExpert, Expert)


def test_dbase_global_instance():
    dbase = DBase.global_instance()
    assert dbase.project_id == "global"
