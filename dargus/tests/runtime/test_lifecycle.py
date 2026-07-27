"""Tests for LifecycleManager."""

import os

from dargus.runtime.context import DargusRuntime
from dargus.runtime.lifecycle import LifecycleManager


def _seed_benchmark_dbase(tmp_path, monkeypatch):
    """Isolated global D-Base (via DARGUS_HOME) with one evidence record."""
    home = str(tmp_path / "dargus_home")
    os.makedirs(home, exist_ok=True)
    monkeypatch.setenv("DARGUS_HOME", home)

    from dargus.dbase import DBase
    from dargus.dbase.manager import DBaseManager

    manager = DBaseManager(DBase.global_instance())
    record = {
        "biological_level": "rct",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {"type": "response_rate", "category": "clinic_efficacy_primary", "value": [0.7]},
        "bg": {"disease_id": ["mondo:0005148"], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/lifecycle"}],
        "source_entry": "10.1234/lifecycle",
        "source_time": "2026-01-01",
    }
    assert manager.write_record(record, dedup=False) is True
    return record["evidence_id"]


class TestLifecycleManager:
    """Tests for LifecycleManager startup/shutdown flow and workflow delegation."""

    def test_constructor_stores_runtime(self):
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        assert lm._runtime is rt

    def test_startup_keeps_runtime_healthy(self):
        """The runtime starts healthy; startup verifies wiring and stays healthy."""
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        rt.embedding_model = object()  # type: ignore[assignment]

        lm = LifecycleManager(rt)
        result = lm.startup()

        assert result is True
        assert rt.healthy is True

    def test_startup_without_models_stays_healthy(self):
        """Missing optional models do not flip the flag at startup."""
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        result = lm.startup()

        assert result is True
        assert rt.healthy is True

    def test_shutdown_sets_unhealthy(self):
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        lm.shutdown()
        assert rt.healthy is False

    def test_shutdown_closes_tool_cache(self):
        rt = DargusRuntime()
        rt.tool_cache.put("heavy", object())
        LifecycleManager(rt).shutdown()
        import pytest

        with pytest.raises(RuntimeError, match="closed"):
            rt.tool_cache.get("heavy")

    def test_startup_then_shutdown_cycle(self):
        rt = DargusRuntime()
        rt.reasoning_llm = object()  # type: ignore[assignment]
        rt.embedding_model = object()  # type: ignore[assignment]

        lm = LifecycleManager(rt)

        assert lm.startup() is True
        assert rt.healthy is True

        lm.shutdown()
        assert rt.healthy is False

    def test_run_predict_delegates_to_workflow(self):
        """LifecycleManager.run_predict delegates to the workflow function."""
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        result = lm.run_predict(
            {
                "workflow": "predict",
                "drug_ids": ["aspirin"],
                "disease_id": "headache",
                "max_rounds": 1,
            }
        )
        assert isinstance(result, dict)
        assert result["workflow"] == "predict"

    def test_run_ingest_delegates_to_workflow(self):
        """LifecycleManager.run_ingest delegates to the workflow function."""
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        result = lm.run_ingest({"workflow": "ingest", "source_path": "/data/test", "max_rounds": 1})
        assert isinstance(result, dict)
        assert result["workflow"] == "ingest"

    def test_run_benchmark_delegates_to_workflow(self, tmp_path, monkeypatch):
        """LifecycleManager.run_benchmark delegates to the workflow function."""
        holdout_id = _seed_benchmark_dbase(tmp_path, monkeypatch)
        rt = DargusRuntime()
        lm = LifecycleManager(rt)
        result = lm.run_benchmark(
            {"workflow": "benchmark", "holdout_ids": [holdout_id], "max_rounds": 1}
        )
        assert isinstance(result, dict)
        assert result["workflow"] == "benchmark"
