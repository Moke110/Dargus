"""Test v0.15.0 biological level vocabulary."""

from dargus.dbase.vocabulary import BIOLOGICAL_LEVELS, READOUT_CATEGORIES


def test_biological_levels_are_11_values():
    assert len(BIOLOGICAL_LEVELS) == 11


def test_rct_epi_in_levels():
    assert "rct" in BIOLOGICAL_LEVELS
    assert "epi" in BIOLOGICAL_LEVELS
    assert "rct-sim" in BIOLOGICAL_LEVELS


def test_clinical_not_in_levels():
    assert "clinical" not in BIOLOGICAL_LEVELS
    assert "clinical-sim" not in BIOLOGICAL_LEVELS


def test_readout_categories_are_23_values():
    assert len(READOUT_CATEGORIES) == 23
    assert "binding" in READOUT_CATEGORIES
    assert "clinic_efficacy_primary" in READOUT_CATEGORIES
