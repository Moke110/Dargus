"""Shared test fixtures for all Dargus tests."""

import pytest


@pytest.fixture
def minimal_dbase(tmp_path):
    """Set up a D-Base with minimal data for testing."""
    import os

    dargus_home = str(tmp_path / "dargus_home")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)
    return dargus_home
