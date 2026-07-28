"""Integration tests for the Ingest workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from dargus.models.reasoning import LLMResponse, LLMUsage, Message, ReasoningLLM
from dargus.workflows.ingest import (
    IngestionReport,
    _build_experts,
    _collect_duplicates,
    _explore_source,
    _extract_evidence,
    _run_ingest,
    run_ingest,
)

# ---------------------------------------------------------------------------
# Fake ReasoningLLM
# ---------------------------------------------------------------------------


class FakeReasoningBackend:
    """ReasoningBackend stub returning controlled evidence dicts per file."""

    def __init__(self, instances_per_file: list[dict[str, Any]] | None = None):
        self._instances = instances_per_file or []
        self._idx = 0

    def chat(self, messages: list[Message], options=None) -> LLMResponse:
        # Return the next instance in the queue, or cycle back
        if self._idx >= len(self._instances):
            self._idx = 0
        instance = self._instances[self._idx] if self._instances else _make_stub_cellular()
        self._idx += 1
        return LLMResponse(
            content=json.dumps(instance),
            usage=LLMUsage(),
            model="fake",
            finish_reason="stop",
        )


def _make_fake_llm(instances: list[dict[str, Any]] | None = None) -> ReasoningLLM:
    return ReasoningLLM(backend=FakeReasoningBackend(instances))


# ---------------------------------------------------------------------------
# Stub evidence factories (matching the 50-field schema shape)
# ---------------------------------------------------------------------------


def _make_stub_molecular() -> dict[str, Any]:
    return {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "sources": [{"rank": 1, "type": "journal", "name": "J Med Chem"}],
        "source_entry": "doi:10.1234/test.mol",
        "source_time": "2024-06-01",
        "xy": {"count": 1},
        "x": {"type": "drug", "value": [{"entity_id": "chembl:118", "entity_label": "aspirin"}]},
        "y": {"type": "binding_affinity", "category": "binding", "value": [8.5]},
        "bg": {"dose_value": 10.0, "dose_unit": "µM"},
    }


def _make_stub_cellular() -> dict[str, Any]:
    return {
        "biological_level": "cellular",
        "evidence_design": "descriptive",
        "sources": [{"rank": 1, "type": "journal", "name": "Cell Rep"}],
        "source_entry": "doi:10.5678/test.cell",
        "source_time": "2024-03-15",
        "xy": {"count": 1},
        "x": {"type": "drug", "value": [{"entity_id": "chembl:118", "entity_label": "aspirin"}]},
        "y": {"type": "viability_72h", "category": "viability", "value": [42.0], "assay": "MTT"},
        "cell_line_id": "cellosaurus:CVCL_0030",
    }


def _make_stub_rct() -> dict[str, Any]:
    return {
        "biological_level": "rct",
        "evidence_design": "two_arm_comparison",
        "sources": [{"rank": 1, "type": "journal", "name": "NEJM"}],
        "source_entry": "doi:10.9999/test.rct",
        "source_time": "2024-01-01",
        "xy": {"count": 2},
        "x": {
            "type": "drug",
            "value": [
                {"entity_id": "chembl:118", "entity_label": "aspirin"},
                {"entity_id": None, "entity_label": "placebo"},
            ],
        },
        "y": {
            "type": "overall_survival",
            "category": "clinic_efficacy_primary",
            "value": [0.82, 0.75],
            "direction": "beneficial",
        },
        "bg": {
            "disease_id": ["mondo:0005249"],
            "drugs": [{"entity_id": "chembl:118", "entity_label": "aspirin"}],
        },
        "clinical_design": {
            "comparator_type": "placebo",
            "blinding": "double",
            "randomized": True,
            "phase": "phase_3",
            "n_arms": 2,
            "population": "adults",
        },
        "is_primary_endpoint": True,
    }


# Expected top-level keys for a structurally valid 50-field evidence dict
_EXPECTED_TOP_KEYS = frozenset(
    {
        "biological_level",
        "evidence_design",
        "sources",
        "source_entry",
        "source_time",
        "xy",
        "x",
        "y",
    }
)


def _has_schema_shape(instance: dict[str, Any]) -> bool:
    """Check that *instance* has the mandatory top-level evidence keys."""
    return _EXPECTED_TOP_KEYS.issubset(instance.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_source_dir() -> str:
    """Create a temp directory with domain-typed files for Explore."""
    with tempfile.TemporaryDirectory() as tmp:
        # Molecular files
        (Path(tmp) / "ligand.sdf").write_text("fake sdf content")
        (Path(tmp) / "compound.smi").write_text("CCO")

        # Biomedical files
        (Path(tmp) / "assay.txt").write_text("cell viability: 42%")

        # Clinical file
        (Path(tmp) / "trial.csv").write_text("patient,outcome\n1,0.82")

        yield tmp


@pytest.fixture
def temp_source_dir_with_bad_file() -> str:
    """Create a temp dir that includes one binary file that will fail extraction."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "ligand.sdf").write_text("fake sdf")
        (Path(tmp) / "assay.txt").write_text("viability data")
        # A binary file — read will succeed but the LLM may produce unparseable output
        (Path(tmp) / "corrupt.bin").write_bytes(b"\x00\x01\x02\x03" * 10)
        yield tmp


# ---------------------------------------------------------------------------
# Explore phase tests
# ---------------------------------------------------------------------------


def test_explore_source_classifies_by_extension(temp_source_dir):
    """Files with known extensions should be assigned to correct domains."""
    batches = _explore_source(temp_source_dir)
    assert "molecule" in batches
    assert "biomedical" in batches
    assert "clinical" in batches
    # .sdf and .smi → molecule
    assert len(batches["molecule"]) == 2
    assert len(batches["biomedical"]) == 1
    assert len(batches["clinical"]) == 1


def test_explore_source_empty_path():
    assert _explore_source("") == {}


def test_explore_source_nonexistent_path():
    assert _explore_source("/nonexistent/path/12345") == {}


def test_explore_source_domain_mapping_override(tmp_path):
    """_domain_mapping overrides extension-based classification."""
    (tmp_path / "data.txt").write_text("some clinical data")
    spec = {"_domain_mapping": {"data.txt": "clinical"}}
    batches = _explore_source(str(tmp_path), spec)
    assert batches == {"clinical": [str(tmp_path / "data.txt")]}


# ---------------------------------------------------------------------------
# Convert phase tests (Expert extraction)
# ---------------------------------------------------------------------------


def test_extract_evidence_stub_mode(tmp_path):
    """Without a fake LLM, experts return stub evidence instances."""
    from dargus.experts.biomed import BiomedExpert

    (tmp_path / "assay.txt").write_text("cell assay data")
    expert = BiomedExpert()  # no reasoning_llm → stub mode
    instances, errors = _extract_evidence(expert, "biomedical", [str(tmp_path / "assay.txt")])

    assert errors == 0
    assert len(instances) == 1
    assert _has_schema_shape(instances[0])
    assert instances[0]["biological_level"] == "cellular"


def test_extract_evidence_fake_llm_produces_instances(tmp_path):
    """Each file fed through a fake LLM yields its configured evidence instances."""
    from dargus.experts.biomed import BiomedExpert

    (tmp_path / "a.txt").write_text("data a")
    (tmp_path / "b.txt").write_text("data b")

    fake_llm = _make_fake_llm([_make_stub_cellular(), _make_stub_cellular()])
    expert = BiomedExpert(reasoning_llm=fake_llm)

    instances, errors = _extract_evidence(
        expert, "biomedical", [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    )

    assert errors == 0
    assert len(instances) == 2
    for inst in instances:
        assert _has_schema_shape(inst)


def test_extract_evidence_bad_file_skipped(tmp_path):
    """A single failing file is logged and skipped; the rest of the batch succeeds."""
    from dargus.experts.biomed import BiomedExpert

    (tmp_path / "good.txt").write_text("good data")
    (tmp_path / "bad.txt").write_text("bad data")

    # Fake LLM that succeeds for good.txt but produces unparseable JSON for bad.txt
    class SelectiveFakeBackend:
        def __init__(self):
            self._call_count = 0

        def chat(self, messages, options=None):
            self._call_count += 1
            if self._call_count == 1:
                # good.txt — valid JSON
                return LLMResponse(
                    content=json.dumps(_make_stub_cellular()),
                    usage=LLMUsage(),
                    model="fake",
                )
            # bad.txt — cause an exception to trigger skip
            raise RuntimeError("simulated extraction failure")

    fake_llm = ReasoningLLM(backend=SelectiveFakeBackend())
    expert = BiomedExpert(reasoning_llm=fake_llm)

    instances, errors = _extract_evidence(
        expert, "biomedical", [str(tmp_path / "good.txt"), str(tmp_path / "bad.txt")]
    )

    # good.txt yields 1 instance; bad.txt triggers skip → 1 error
    assert len(instances) == 1
    assert errors == 1
    assert _has_schema_shape(instances[0])


# ---------------------------------------------------------------------------
# End-to-end test: run_ingest with real temp dir + fake LLM
# ---------------------------------------------------------------------------


def test_run_ingest_end_to_end_with_fake_llm(tmp_path):
    """Full ingest (Explore + Convert) with a real temp dir and fake LLM.

    - Creates domain-typed files across molecule/biomedical/clinical domains.
    - Each Expert's extract() is invoked with the fake LLM returning
      structured evidence dicts covering the 50-field schema shape.
    - Asserts the structured instances are produced per domain with the
      expected schema keys.
    """
    # Create files across domains
    (tmp_path / "ligand.sdf").write_text("fake sdf")
    (tmp_path / "compound.smi").write_text("smiles: CCO")
    (tmp_path / "assay.txt").write_text("cell viability 42%")
    (tmp_path / "trial.csv").write_text("patient_id,outcome\n1,improved")

    # Build a fake LLM that returns domain-appropriate instances
    fake_instances = [
        _make_stub_molecular(),  # for molecule.sdf
        _make_stub_molecular(),  # for compound.smi
        _make_stub_cellular(),  # for assay.txt
        _make_stub_rct(),  # for trial.csv
    ]
    fake_llm = _make_fake_llm(fake_instances)

    spec: dict[str, Any] = {
        "workflow": "ingest",
        "source_path": str(tmp_path),
        "max_rounds": 5,
        "_reasoning_llm": fake_llm,
    }

    result = run_ingest(spec)
    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"
    assert result["status"] in ("completed", "converged")
    assert result["n_records"] == 4  # one per file
    assert result["n_errors"] == 0

    # Check session recorded explore_batches and evidence_instances
    session = result["session"]
    assert "explore_batches" in session
    assert "evidence_instances" in session

    # Verify instances are per-domain
    explore_batches = session["explore_batches"]
    assert explore_batches.keys() >= {"molecule", "biomedical", "clinical"}

    instances = session["evidence_instances"]
    assert len(instances) == 4
    for inst in instances:
        assert _has_schema_shape(inst), f"missing schema keys in {list(inst.keys())}"

    # Verify domain diversity in produced instances
    levels = {inst.get("biological_level") for inst in instances}
    assert "molecular" in levels
    assert "cellular" in levels or "rct" in levels  # at least one non-mol

    # Verify session rounds recorded per-domain extraction
    rounds = session.get("rounds", [])
    domains_seen = {r["domain"] for r in rounds}
    assert domains_seen >= {"molecule", "biomedical", "clinical"}


def test_run_ingest_bad_file_skipped(tmp_path):
    """A single failing file is logged and skipped; other files still produce instances."""
    (tmp_path / "good.smi").write_text("CCO")
    (tmp_path / "bad.smi").write_text("bad data")

    # Fake LLM: succeeds for first call, raises for second
    class FailOnSecondBackend:
        def __init__(self):
            self._count = 0

        def chat(self, messages, options=None):
            self._count += 1
            if self._count == 1:
                return LLMResponse(
                    content=json.dumps(_make_stub_molecular()),
                    usage=LLMUsage(),
                    model="fake",
                )
            raise RuntimeError("extraction failure on second file")

    fake_llm = ReasoningLLM(backend=FailOnSecondBackend())

    spec: dict[str, Any] = {
        "workflow": "ingest",
        "source_path": str(tmp_path),
        "max_rounds": 5,
        "_reasoning_llm": fake_llm,
    }

    result = run_ingest(spec)
    assert result["n_records"] == 1
    assert result["n_errors"] == 1
    assert _has_schema_shape(result["session"]["evidence_instances"][0])


def test_run_ingest_no_source():
    """With empty source_path, returns zero records."""
    result = run_ingest({"workflow": "ingest", "source_path": "", "max_rounds": 1})
    assert result["n_records"] == 0
    assert result["n_errors"] == 0


# ---------------------------------------------------------------------------
# Stub mode end-to-end test (no fake LLM wired)
# ---------------------------------------------------------------------------


def test_run_ingest_stub_mode(tmp_path):
    """When no LLM is wired, each Expert returns one stub per domain."""
    (tmp_path / "ligand.sdf").write_text("mol data")
    (tmp_path / "assay.txt").write_text("cell data")
    (tmp_path / "trial.csv").write_text("patient data")

    spec: dict[str, Any] = {
        "workflow": "ingest",
        "source_path": str(tmp_path),
        "max_rounds": 5,
    }
    result = run_ingest(spec)

    # Stub mode: one instance per domain (= per batch with files)
    assert result["n_records"] >= 1
    assert result["n_errors"] == 0

    # Each instance should have the schema shape
    for inst in result["session"]["evidence_instances"]:
        assert _has_schema_shape(inst)


# ---------------------------------------------------------------------------
# Duplicate review and gate tests
# ---------------------------------------------------------------------------


def test_collect_duplicates_empty_for_stub():
    records = [{"id": "r1"}, {"id": "r2"}]
    dups = _collect_duplicates(records)
    assert dups == []


def test_collect_duplicates_injectable_via_task_spec():
    """I4: _collect_duplicates returns injected duplicates from task_spec."""
    records = [{"id": "r1"}]
    fake_dups = [
        {"evidence_id": "dup-1", "reason": "similar_fingerprint"},
        {"evidence_id": "dup-2", "reason": "exact_match"},
    ]
    dups = _collect_duplicates(records, task_spec={"_duplicate_records": fake_dups})
    assert len(dups) == 2
    assert dups[0]["evidence_id"] == "dup-1"


def test_ingest_duplicate_review_path_reached():
    """I4: When duplicates are injected, the n_duplicates > 0 branch is executed."""
    spec = {
        "workflow": "ingest",
        "source_path": "",
        "max_rounds": 1,
        "_duplicate_records": [{"evidence_id": "dup-001", "reason": "exact_match"}],
    }
    result = _run_ingest(spec)
    assert result["n_duplicates"] == 1
    confirmations = result["session"].get("confirmations", [])
    assert len(confirmations) == 1
    assert confirmations[0]["type"] == "duplicate_review"
    assert confirmations[0]["n_duplicates"] == 1


def test_run_ingest_handles_duplicate_review(tmp_path):
    """Duplicate review gate should store confirmation info."""
    (tmp_path / "assay.txt").write_text("data")
    spec: dict[str, Any] = {
        "workflow": "ingest",
        "source_path": str(tmp_path),
        "max_rounds": 5,
        "require_confirmation": True,
    }
    result = run_ingest(spec)
    assert "session" in result


# ---------------------------------------------------------------------------
# Backward-compat dataclasses
# ---------------------------------------------------------------------------


def test_ingestion_report_defaults():
    r = IngestionReport()
    assert r.n_records == 0
    assert r.n_skipped == 0
    assert r.dbase_size == 0
    assert r.errors == []


def test_training_report_is_ingestion_report():
    from dargus.workflows.ingest import TrainingReport

    r = TrainingReport(n_records=10)
    assert isinstance(r, IngestionReport)
    assert r.n_records == 10


# ---------------------------------------------------------------------------
# Backward-compat run_ingest(datadir) signature tests
# ---------------------------------------------------------------------------


def test_run_ingest_backward_compat_datadir_string(tmp_path):
    """C2: run_ingest(datadir) returns IngestionReport for backward compat."""
    (tmp_path / "assay.txt").write_text("cell viability data")
    result = run_ingest(str(tmp_path))
    assert isinstance(result, IngestionReport)
    assert result.n_records >= 1


def test_run_ingest_backward_compat_with_reset(tmp_path):
    """C2: run_ingest(datadir, reset=True) works."""
    (tmp_path / "assay.txt").write_text("cell viability data")
    result = run_ingest(str(tmp_path), reset=True)
    assert isinstance(result, IngestionReport)


def test_run_ingest_backward_compat_with_disease_kb_dir(tmp_path):
    """C2: run_ingest(datadir, disease_kb_dir=...) works."""
    (tmp_path / "assay.txt").write_text("cell viability data")
    result = run_ingest(str(tmp_path), disease_kb_dir="/data/kb")
    assert isinstance(result, IngestionReport)


def test_run_ingest_new_api_dict(tmp_path):
    """C2: run_ingest(task_spec) still returns dict."""
    (tmp_path / "assay.txt").write_text("data")
    result = run_ingest({"workflow": "ingest", "source_path": str(tmp_path), "max_rounds": 1})
    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"


def test_run_ingest_with_max_rounds(tmp_path):
    """With few rounds, should still complete."""
    (tmp_path / "assay.txt").write_text("data")
    spec = {"workflow": "ingest", "source_path": str(tmp_path), "max_rounds": 2}
    result = run_ingest(spec)
    assert result["n_records"] >= 1


# ---------------------------------------------------------------------------
# _build_experts tests
# ---------------------------------------------------------------------------


def test_build_experts_returns_all_domains():
    experts = _build_experts()
    assert set(experts.keys()) == {"molecule", "biomedical", "bioinformatics", "clinical"}


def test_build_experts_with_fake_llm():
    fake_llm = _make_fake_llm()
    experts = _build_experts(reasoning_llm=fake_llm)
    for expert in experts.values():
        assert expert._reasoning_llm is fake_llm
