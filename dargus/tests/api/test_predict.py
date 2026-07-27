"""Contract tests for dargus.api facade — v0.15.0 evidence dict API."""

import pytest

import dargus


@pytest.fixture
def minimal_dbase(tmp_path):
    """Set up a D-Base with minimal data for testing."""
    import os

    dargus_home = str(tmp_path / "dargus_home")
    os.environ["DARGUS_HOME"] = dargus_home
    os.makedirs(dargus_home, exist_ok=True)
    return dargus_home


def test_api_predict_returns_prediction_matrix(minimal_dbase):
    """dargus.predict() returns the correct output contract shape."""
    result = dargus.predict(
        drug_ids=["aspirin"],
        disease_id="headache",
        endpoints=["efficacy"],
    )
    assert isinstance(result, dict)
    assert "aspirin" in result
    assert "headache" in result["aspirin"]
    assert "efficacy" in result["aspirin"]["headache"]
    entry = result["aspirin"]["headache"]["efficacy"]
    assert "efficacy_score" in entry
    assert "confidence_score" in entry
    assert "supporting_records" in entry
    assert "reasoning_mode" in entry
    assert "confidence_level" in entry
    if entry["confidence_level"] == "insufficient_data":
        assert entry["efficacy_score"] is None
        assert entry["confidence_score"] is None
    else:
        assert 0.0 <= entry["efficacy_score"] <= 1.0
        assert 0.0 <= entry["confidence_score"] <= 1.0


def test_api_ingest_returns_ingestion_report(minimal_dbase, tmp_path):
    """dargus.ingest() returns an IngestionReport."""
    datadir = tmp_path / "test_data"
    datadir.mkdir()
    report = dargus.ingest(datadir=str(datadir))
    assert hasattr(report, "n_records")
    assert hasattr(report, "n_skipped")
    assert hasattr(report, "dbase_size")


def test_api_query_dbase_returns_list(minimal_dbase):
    """dargus.query_dbase() returns a list."""
    records = dargus.query_dbase(disease_id="headache")
    assert isinstance(records, list)


def test_api_status_returns_dict(minimal_dbase):
    """dargus.status() returns a dict with required keys."""
    status = dargus.status()
    assert isinstance(status, dict)
    assert "dargus_home" in status
    assert "n_records" in status
    assert "working_dbase" in status


def test_api_benchmark_validates_config(minimal_dbase):
    """dargus.benchmark() aborts when the holdout selection matches zero records."""
    import dargus

    with pytest.raises(ValueError, match="zero records"):
        dargus.benchmark(strip={"source.type": "benchmark"})


def test_api_query_expert_returns_stub():
    """dargus.query_expert() returns a stub dict with expert name."""
    import dargus

    result = dargus.query_expert("MoleculeExpert")
    assert isinstance(result, dict)
    assert result["expert"] == "MoleculeExpert"
    assert "note" in result
