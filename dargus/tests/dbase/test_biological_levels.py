"""Test biological level vocabulary."""

from dargus.dbase.vocabulary import VocabularyManager


def test_biological_levels_are_11_values():
    vm = VocabularyManager()
    levels = vm.get_enum_values("biological_level")
    assert len(levels) == 11


def test_rct_epi_in_levels():
    vm = VocabularyManager()
    levels = vm.get_enum_values("biological_level")
    assert "rct" in levels
    assert "epi" in levels
    assert "rct-sim" in levels


def test_clinical_not_in_levels():
    vm = VocabularyManager()
    levels = vm.get_enum_values("biological_level")
    assert "clinical" not in levels
    assert "clinical-sim" not in levels


def test_readout_categories_are_23_values():
    vm = VocabularyManager()
    categories = vm.get_enum_values("y_category")
    assert len(categories) == 23
    assert "binding" in categories
    assert "clinic_efficacy_primary" in categories
