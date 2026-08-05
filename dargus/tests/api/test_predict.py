"""Contract tests for the dargus.api facade."""

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


def test_runtime_reuse_bootstraps_once():
    """SPEC-B: two consecutive API calls reuse the same DargusRuntime."""
    import dargus.api as api

    api._RUNTIME_CACHE = None
    r1 = api._get_runtime()
    r2 = api._get_runtime()
    assert r1 is r2
    # Reset the process cache so other tests bootstrap their own runtime.
    api._RUNTIME_CACHE = None


def test_runtime_reuse_bootstrap_failure_refuses_session(monkeypatch):
    """SPEC-B: a bootstrap failure marks the runtime unhealthy and entry
    points refuse new sessions — no silent fallback."""
    import dargus.api as api

    api._RUNTIME_CACHE = None

    def _fail_bootstrap(*_a, **_k):
        raise RuntimeError("config missing")

    import importlib

    bootstrap_mod = importlib.import_module("dargus.runtime.bootstrap")

    monkeypatch.setattr(bootstrap_mod, "bootstrap", _fail_bootstrap)
    with pytest.raises(RuntimeError, match="refusing new session"):
        api._get_runtime()
    # A failed bootstrap leaves the cache empty so a later healthy call can retry.
    assert api._RUNTIME_CACHE is None


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


def test_api_ingest_returns_dict(minimal_dbase, tmp_path):
    """dargus.ingest() returns a result dict."""
    datadir = tmp_path / "test_data"
    datadir.mkdir()
    report = dargus.ingest(datadir=str(datadir))
    assert isinstance(report, dict)
    assert "n_records" in report


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


def test_api_query_expert_returns_stub():
    """dargus.query_expert() returns a stub dict with expert name."""
    import dargus

    result = dargus.query_expert("MoleculeExpert")
    assert isinstance(result, dict)
    assert result["expert"] == "MoleculeExpert"
    assert "note" in result
