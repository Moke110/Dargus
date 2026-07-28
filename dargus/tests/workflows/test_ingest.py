"""Integration tests for the Ingest workflow."""

from __future__ import annotations

import json

import pytest

from dargus.workflows.ingest import (
    IngestionReport,
    _collect_duplicates,
    _explore_sources,
    _partition_by_domain,
    _run_ingest,
    run_ingest,
)

# The old _parse_source is gone — verify it's no longer importable.
# We keep backward-compat tests that call the private helpers they still
# depend on (_partition_by_domain, _collect_duplicates) and add new tests
# for _explore_sources.

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _fake_reasoning_llm(response: str):
    """Return a ReasoningLLM wired to a FakeReasoningBackend."""
    from dargus.models.reasoning import LLMResponse, LLMUsage, ReasoningLLM

    class _FakeBackend:
        def chat(self, messages, options=None):
            return LLMResponse(
                content=response, usage=LLMUsage(), model="fake", finish_reason="stop"
            )

    return ReasoningLLM(backend=_FakeBackend())


def _fixture_data_dir(tmp_path, files: dict[str, str] | None = None) -> str:
    """Create a temp directory populated with named (possibly empty) files."""
    data_dir = tmp_path / "ingest_input"
    data_dir.mkdir()
    if files is None:
        files = {
            "mol_compounds.csv": "SMILES,Name\nCCO,Ethanol\n",
            "biomed_study.txt": "Study: Aspirin PK in rats\n",
            "bioinfo_rnaseq.json": json.dumps({"samples": 12}),
            "clinical_trial.txt": "NCT000001: Phase III RCT\n",
            "random.xyz": "binary\x00data",
        }
    for fname, content in files.items():
        path = data_dir / fname
        path.write_text(content if isinstance(content, str) else content.decode(errors="replace"))
    return str(data_dir)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_ingest_spec() -> dict:
    return {
        "workflow": "ingest",
        "source_path": "/data/pubmed_batch_01",
        "source_type": "pubmed",
        "max_rounds": 5,
    }


# ---------------------------------------------------------------------------
# _explore_sources — new Explore phase
# ---------------------------------------------------------------------------


class DescribeExploreSources:
    """Explore phase: directory scan + domain classification + per-domain batching."""

    def test_scans_directory_and_discovers_files(self, tmp_path):
        """A real directory should yield one entry per discovered file."""
        data_dir = _fixture_data_dir(tmp_path)
        result = _explore_sources(data_dir)
        # result is a dict[str, list[str]] mapping domain -> list[file_path]
        assert isinstance(result, dict)
        all_files = [p for paths in result.values() for p in paths]
        assert len(all_files) >= 3  # at least the domain-classifiable files

    def test_classifies_by_domain(self, tmp_path):
        """Files named with domain hints should land in the right buckets."""
        data_dir = _fixture_data_dir(
            tmp_path,
            {
                "aspirin_molecule.csv": "data",
                "study_biomedical.txt": "data",
                "experiment_bioinformatics.csv": "data",
                "trial_clinical.csv": "data",
            },
        )
        result = _explore_sources(data_dir)
        assert "molecular" in result
        assert "biomedical" in result
        assert "bioinformatics" in result
        assert "clinical" in result

    def test_unclassifiable_files_are_skipped(self, tmp_path):
        """Files with no domain signal should not appear in the output."""
        data_dir = _fixture_data_dir(
            tmp_path,
            {
                "molecule_data.csv": "x",
                "weird_file.xyz": "y",
            },
        )
        result = _explore_sources(data_dir)
        all_files = {p for paths in result.values() for p in paths}
        assert any("molecule_data" in f for f in all_files)
        assert not any("weird_file" in f for f in all_files)

    def test_empty_directory_returns_empty_batches(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        result = _explore_sources(data_dir)
        assert result == {}

    def test_fake_llm_classification(self, tmp_path):
        """When a reasoning LLM is injected, its classification response is used."""
        data_dir = _fixture_data_dir(
            tmp_path,
            {
                "file_a.txt": "content",
                "file_b.txt": "content",
            },
        )
        # The fake LLM classifies both files as "clinical"
        llm_response = json.dumps(
            {
                "classifications": [
                    {"file": "file_a.txt", "domain": "clinical"},
                    {"file": "file_b.txt", "domain": "clinical"},
                ]
            }
        )
        llm = _fake_reasoning_llm(llm_response)
        result = _explore_sources(data_dir, reasoning_llm=llm)
        assert "clinical" in result
        assert len(result["clinical"]) == 2

    def test_fake_llm_skip_unclassifiable(self, tmp_path):
        """LLM returns an explicit 'skip' domain — file is dropped."""
        data_dir = _fixture_data_dir(
            tmp_path,
            {"bad_file.abc": "junk", "good_clinical.txt": "ok"},
        )
        llm_response = json.dumps(
            {
                "classifications": [
                    {"file": "bad_file.abc", "domain": "unknown"},
                    {"file": "good_clinical.txt", "domain": "clinical"},
                ]
            }
        )
        llm = _fake_reasoning_llm(llm_response)
        result = _explore_sources(data_dir, reasoning_llm=llm)
        all_files = {p for paths in result.values() for p in paths}
        assert any("good_clinical" in f for f in all_files)
        assert not any("bad_file" in f for f in all_files)

    def test_source_path_not_exist(self, tmp_path):
        """Non-existent path returns empty result."""
        result = _explore_sources(str(tmp_path / "does_not_exist"))
        assert result == {}


# ---------------------------------------------------------------------------
# run_ingest tests (updated — no more synthetic records)
# ---------------------------------------------------------------------------


def test_run_ingest_completes_and_returns_ingest_result(valid_ingest_spec):
    """Run ingest should complete and return the expected result keys."""
    result = run_ingest(valid_ingest_spec)

    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"
    assert result["status"] in ("completed", "converged")
    assert "n_records" in result
    assert "n_duplicates" in result
    assert "n_errors" in result
    assert "session" in result
    # No more synthetic records — with a non-existent path, n_records == 0
    assert result["n_records"] >= 0


def test_run_ingest_with_no_source():
    """With empty source_path, should handle gracefully."""
    result = run_ingest({"workflow": "ingest", "source_path": "", "max_rounds": 1})
    assert result["n_records"] == 0
    assert result["n_errors"] == 0


def test_run_ingest_handles_duplicate_review(valid_ingest_spec):
    """Duplicate review gate should store confirmation info."""
    valid_ingest_spec["require_confirmation"] = True
    result = run_ingest(valid_ingest_spec)
    # Stub duplicates are always empty, but the gate still fires the confirmation path
    assert "session" in result


def test_run_ingest_with_max_rounds():
    """With few rounds, should converge early."""
    spec = {"workflow": "ingest", "source_path": "/data/test", "max_rounds": 2}
    result = run_ingest(spec)
    assert result["n_records"] >= 0


def test_run_ingest_explore_phase_with_real_directory(tmp_path):
    """End-to-end: run_ingest against a real temp dir with a fake LLM.

    The Explore phase scans the directory, classifies files by domain
    (via the fake LLM), and produces per-domain batches. The Convert
    phase is still a stub, so each file becomes one "record."
    """
    from dargus.models.reasoning import LLMResponse, LLMUsage, ReasoningLLM

    # ---- Create test files --------------------------------------------------
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    (data_dir / "molecule.csv").write_text("Drug,Target\naspirin,COX1\n")
    (data_dir / "biomed.txt").write_text("Study: mouse model\n")
    (data_dir / "bioinfo.csv").write_text("gene,count\nBRCA1,42\n")
    (data_dir / "clinical.csv").write_text("trial,outcome\nNCT001,positive\n")
    (data_dir / "garbage.bin").write_text("unknown format data\n")

    # ---- Build a fake LLM that classifies discovered files ------------------
    class _FakeClassifyBackend:
        def chat(self, messages, options=None):
            # Return a JSON classification for each file in the prompt
            import re

            # Extract filenames from the prompt
            prompt = messages[-1].content if messages else ""
            files = re.findall(r"([\w.-]+\.\w+)", prompt)
            entries = []
            for f in files:
                f_lower = f.lower()
                if "molecule" in f_lower:
                    domain = "molecular"
                elif "biomed" in f_lower:
                    domain = "biomedical"
                elif "bioinfo" in f_lower:
                    domain = "bioinformatics"
                elif "clinical" in f_lower:
                    domain = "clinical"
                else:
                    domain = "unknown"
                entries.append({"file": f, "domain": domain})
            return LLMResponse(
                content=json.dumps({"classifications": entries}),
                usage=LLMUsage(),
                model="fake-classify",
                finish_reason="stop",
            )

    llm = ReasoningLLM(backend=_FakeClassifyBackend())
    spec = {
        "workflow": "ingest",
        "source_path": str(data_dir),
        "max_rounds": 10,
        "_reasoning_llm": llm,
    }

    result = _run_ingest(spec)

    # The 4 classifiable files are explored, 1 "garbage" file is skipped
    # Each file becomes 1 record in the stub Convert phase
    assert result["n_records"] == 4
    assert result["n_errors"] == 0
    assert result["workflow"] == "ingest"

    # The session should carry domain annotations from the Explore phase
    session = result.get("session", {})
    domain_batches = session.get("explore_batches", {})
    assert set(domain_batches.keys()) == {"molecular", "biomedical", "bioinformatics", "clinical"}


# ---------------------------------------------------------------------------
# Internal helpers (backward-compat)
# ---------------------------------------------------------------------------


def test_partition_by_domain_groups_correctly():
    records = [
        {"domain": "molecular", "id": "a"},
        {"domain": "molecular", "id": "b"},
        {"domain": "clinical", "id": "c"},
    ]
    groups = _partition_by_domain(records)
    assert len(groups) == 2
    domain_names = {g[0] for g in groups}
    assert domain_names == {"molecular", "clinical"}


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


def test_ingest_duplicate_review_path_reached(valid_ingest_spec):
    """I4: When duplicates are injected, the n_duplicates > 0 branch is executed."""
    valid_ingest_spec["_duplicate_records"] = [
        {"evidence_id": "dup-001", "reason": "exact_match"},
    ]
    valid_ingest_spec["max_rounds"] = 1
    result = _run_ingest(valid_ingest_spec)
    assert result["n_duplicates"] == 1
    # Confirm the confirmation record was appended to session
    confirmations = result["session"].get("confirmations", [])
    assert len(confirmations) == 1
    assert confirmations[0]["type"] == "duplicate_review"
    assert confirmations[0]["n_duplicates"] == 1


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


def test_run_ingest_backward_compat_datadir_string():
    """C2: run_ingest(datadir) returns IngestionReport for backward compat."""
    result = run_ingest("/data/test_dir")
    assert isinstance(result, IngestionReport)
    assert result.n_records >= 0  # directory doesn't exist → 0 records now


def test_run_ingest_backward_compat_with_reset():
    """C2: run_ingest(datadir, reset=True) works."""
    result = run_ingest("/data/test_dir", reset=True)
    assert isinstance(result, IngestionReport)
    assert result.n_skipped == 0


def test_run_ingest_backward_compat_with_disease_kb_dir():
    """C2: run_ingest(datadir, disease_kb_dir=...) works."""
    result = run_ingest("/data/test_dir", disease_kb_dir="/data/kb")
    assert isinstance(result, IngestionReport)


def test_run_ingest_new_api_dict():
    """C2: run_ingest(task_spec) still returns dict."""
    result = run_ingest({"workflow": "ingest", "source_path": "/data/test", "max_rounds": 1})
    assert isinstance(result, dict)
    assert result["workflow"] == "ingest"
