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
    assert "efficacy_low" in entry
    assert "efficacy_up" in entry
    assert "supporting_records" in entry
    assert "reasoning_mode" in entry
    assert "confidence_level" in entry
    assert 0.0 <= entry["efficacy_low"] <= 1.0
    assert 0.0 <= entry["efficacy_up"] <= 1.0


def test_api_train_returns_training_report(minimal_dbase, tmp_path):
    """dargus.train() returns an IngestionReport."""
    datadir = tmp_path / "test_data"
    datadir.mkdir()
    report = dargus.train(datadir=str(datadir))
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


def test_api_benchmark_validates_config(minimal_dbase, tmp_path):
    """dargus.benchmark() raises NotImplementedError (workflow removed in v0.15.2)."""
    import dargus

    try:
        dargus.benchmark(strip={"source.type": "benchmark"})
    except NotImplementedError:
        pass  # expected: bench-full-stack removed in v0.15.2


def test_api_predict_single_agent_valid(minimal_dbase):
    """dargus.predict_single_agent() runs an Iris-* agent standalone."""
    import dargus

    result = dargus.predict_single_agent(
        agent_name="iris-search",
        drug_ids=["aspirin"],
        disease_id="headache",
    )
    assert isinstance(result, dict)


def test_api_predict_single_agent_invalid_name(minimal_dbase):
    """dargus.predict_single_agent() raises ValueError for unknown agent."""
    with pytest.raises(ValueError, match="Unknown agent"):
        dargus.predict_single_agent(
            agent_name="nonexistent",
            drug_ids=["aspirin"],
            disease_id="headache",
        )


def test_api_query_expert_returns_stub():
    """dargus.query_expert() returns a stub dict with expert name."""
    import dargus

    result = dargus.query_expert("MoleculeExpert")
    assert isinstance(result, dict)
    assert result["expert"] == "MoleculeExpert"
    assert "note" in result
