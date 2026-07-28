"""Tests for VocabularyManager — CURIE registry + enums + gated add_term."""

import tempfile
from pathlib import Path

import pytest

from dargus.dbase import validate as val_mod
from dargus.dbase.vocabulary import VocabularyManager


def test_vocab_enum_values():
    vm = VocabularyManager()
    assert "rct" in vm.get_enum("biological_level")
    assert "epi" in vm.get_enum("biological_level")
    assert "clinical" not in vm.get_enum("biological_level")
    assert len(vm.get_enum("biological_level")) == 11


def test_vocab_curie_validation():
    vm = VocabularyManager()
    assert vm.validate_curie("chembl:CHEMBL25")
    assert vm.validate_curie("uniprot:P35354")
    assert vm.validate_curie("mondo:0005180")
    assert not vm.validate_curie("chembl:bad_format")
    assert not vm.validate_curie("unknown:acc")


def test_vocab_save_load():
    vm = VocabularyManager()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        vm.save(path)
        vm2 = VocabularyManager.load(path)
        assert "rct" in vm2.get_enum("biological_level")


# ── gated add_term tests (S8_T1) ───────────────────────────────────────────────


class TestGatedAddTerm:
    """TDD suite for VocabularyManager gated add_term."""

    @staticmethod
    def _make_vm(tmp_path, extra_data=None, pending_file_name="pending.json") -> VocabularyManager:
        """Create a VocabularyManager with a custom vocab loaded and temp pending path."""
        pending_path = tmp_path / pending_file_name
        vm = VocabularyManager(pending_path=pending_path)
        if extra_data:
            vm._data.update(extra_data)
        return vm

    # ── propose ────────────────────────────────────────────────────────────────

    def test_add_term_stages_pending_with_provenance(self, tmp_path):
        """add_term stages a term as pending; provenance recorded."""
        vm = self._make_vm(tmp_path)
        vm.add_term(
            "evidence_design",
            "case_only",
            provenance={"who": "researcher-a", "why": "new design type"},
        )
        pending = vm.pending_terms()
        assert "evidence_design" in pending
        pending_designs = pending["evidence_design"]
        assert len(pending_designs) == 1
        assert pending_designs[0]["term"] == "case_only"
        assert pending_designs[0]["provenance"]["who"] == "researcher-a"
        assert "when" in pending_designs[0]["provenance"]

    def test_pending_term_not_in_active_enum(self, tmp_path):
        """A staged pending term is NOT returned by get_enum_values."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        assert "case_only" not in vm.get_enum("evidence_design")

    def test_pending_term_not_valid(self, tmp_path):
        """A staged pending term fails is_valid_enum_value."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        assert not vm.is_valid_enum_value("evidence_design", "case_only")

    def test_pending_term_rejected_by_validator(self, tmp_path):
        """validate_evidence rejects a record using a pending-only term."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")

        # Simulate the vocab cache: VM's active data does NOT contain the pending term.
        # The validator reads the vocab cache — so it will reject the term.
        val_mod._vocab = vm.to_dict()
        try:
            result = val_mod.validate_evidence(
                {
                    "biological_level": "rct",
                    "is_clinical": 1,
                    "evidence_design": "case_only",
                    "sources": [{"rank": 1, "type": "journal", "name": "Test"}],
                    "source_entry": "PMID:123456",
                    "source_time": "2024-01-01",
                    "bg": {"disease_id": ["mondo:0005180"]},
                    "x": {"type": "drug", "value": [{"entity_id": "chembl:CHEMBL25"}]},
                    "y": {"type": "ORR", "value": [0.5], "category": "clinic_efficacy_primary"},
                    "xy": {"count": 1},
                }
            )
            assert not result.ok
            assert any(
                "case_only" in err for err in result.hard_errors
            ), f"Expected validation failure for pending term; got: {result.hard_errors}"
        finally:
            val_mod._vocab = {}

    # ── approve ────────────────────────────────────────────────────────────────

    def test_approve_moves_term_from_pending_to_active(self, tmp_path):
        """After approval, term appears in get_enum_values."""
        vm = self._make_vm(
            tmp_path,
            extra_data={
                "evidence_design": {
                    "description": "Evidence structure type.",
                    "values": ["two_arm_comparison"],
                }
            },
        )
        vm.add_term("evidence_design", "case_only")
        vm.approve_term("evidence_design", "case_only", approved_by="reviewer-1")

        assert "case_only" in vm.get_enum("evidence_design")
        assert vm.is_valid_enum_value("evidence_design", "case_only")
        assert "evidence_design" not in vm.pending_terms()

    def test_approve_requires_approver_identity(self, tmp_path):
        """approve_term requires approved_by to be a non-empty string."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        with pytest.raises(ValueError, match="approved_by"):
            vm.approve_term("evidence_design", "case_only", approved_by="")

    def test_approve_validates(self, tmp_path):
        """After approval, validate_evidence accepts the term."""
        vm = self._make_vm(
            tmp_path,
            extra_data={
                "evidence_design": {
                    "description": "Evidence structure type.",
                    "values": ["two_arm_comparison"],
                }
            },
        )
        vm.add_term("evidence_design", "case_only")
        vm.approve_term("evidence_design", "case_only", approved_by="reviewer-1")

        # Feed VM's active data into the validator's module-level cache
        val_mod._vocab = vm.to_dict()
        try:
            result = val_mod.validate_evidence(
                {
                    "biological_level": "rct",
                    "is_clinical": 1,
                    "evidence_design": "case_only",
                    "sources": [{"rank": 1, "type": "journal", "name": "Test"}],
                    "source_entry": "PMID:123456",
                    "source_time": "2024-01-01",
                    "bg": {"disease_id": ["mondo:0005180"]},
                    "x": {"type": "drug", "value": [{"entity_id": "chembl:CHEMBL25"}]},
                    "y": {"type": "ORR", "value": [0.5], "category": "clinic_efficacy_primary"},
                    "xy": {"count": 1},
                }
            )
            assert result.ok, (
                f"Expected validation pass after approval;" f" errors: {result.hard_errors}"
            )
        finally:
            val_mod._vocab = {}

    # ── reject ─────────────────────────────────────────────────────────────────

    def test_reject_discards_pending_term(self, tmp_path):
        """Rejection removes the term from pending and it stays absent from active."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        assert "evidence_design" in vm.pending_terms()

        vm.reject_term("evidence_design", "case_only")

        assert "case_only" not in vm.get_enum("evidence_design")
        assert "evidence_design" not in vm.pending_terms()

    def test_reject_nonexistent_is_noop(self, tmp_path):
        """Rejecting a term that doesn't exist in pending is a no-op."""
        vm = self._make_vm(tmp_path)
        vm.reject_term("evidence_design", "nonexistent")
        # no exception raised

    # ── persistence ────────────────────────────────────────────────────────────

    def test_pending_survives_save_load_cycle(self, tmp_path):
        """Pending terms persist through save/load."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "test_state.json"
            vm.save_state(state_path)

            vm2 = VocabularyManager.load_state(state_path)
            pending = vm2.pending_terms()
            assert "evidence_design" in pending
            assert pending["evidence_design"][0]["term"] == "case_only"

    def test_pending_survives_restart(self, tmp_path):
        """After loading from state, pending terms are still pending (not active)."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "test_state.json"
            vm.save_state(state_path)

            vm2 = VocabularyManager.load_state(state_path)
            assert "case_only" not in vm2.get_enum("evidence_design")
            assert not vm2.is_valid_enum_value("evidence_design", "case_only")
            assert "evidence_design" in vm2.pending_terms()

    def test_approval_clears_pending_in_persisted_state(self, tmp_path):
        """After approval, persisted state has term active and pending cleared."""
        vm = self._make_vm(
            tmp_path,
            extra_data={
                "evidence_design": {
                    "description": "Evidence structure type.",
                    "values": ["two_arm_comparison"],
                }
            },
        )
        vm.add_term("evidence_design", "case_only")
        vm.approve_term("evidence_design", "case_only", approved_by="reviewer-1")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "test_state.json"
            vm.save_state(state_path)

            vm2 = VocabularyManager.load_state(state_path)
            assert "case_only" in vm2.get_enum("evidence_design")
            assert "evidence_design" not in vm2.pending_terms()

    # ── edge cases ─────────────────────────────────────────────────────────────

    def test_add_term_duplicate_active_is_noop(self, tmp_path):
        """Adding a term already in the active set is a no-op (no duplicate pending)."""
        vm = self._make_vm(tmp_path)
        # 'rct' is already active
        vm.add_term("biological_level", "rct")
        assert "biological_level" not in vm.pending_terms()

    def test_add_term_duplicate_pending_is_noop(self, tmp_path):
        """Adding the same term twice only records it once."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        vm.add_term("evidence_design", "case_only")
        assert len(vm.pending_terms()["evidence_design"]) == 1

    def test_approve_nonexistent_term_raises(self, tmp_path):
        """Approving a term not in pending raises ValueError."""
        vm = self._make_vm(tmp_path)
        with pytest.raises(ValueError, match="not in pending"):
            vm.approve_term("evidence_design", "nonexistent", approved_by="reviewer-1")

    def test_add_term_unknown_vocabulary_creates_entry(self, tmp_path):
        """Adding a term to a vocabulary not yet present creates a pending entry."""
        vm = self._make_vm(tmp_path)
        vm.add_term("novel_vocab", "novel_term")
        assert vm.pending_terms()["novel_vocab"][0]["term"] == "novel_term"

    def test_add_term_without_provenance_uses_defaults(self, tmp_path):
        """add_term called without provenance fills in sensible defaults."""
        vm = self._make_vm(tmp_path)
        vm.add_term("evidence_design", "case_only")
        p = vm.pending_terms()["evidence_design"][0]
        assert p["provenance"]["who"] == "unknown"
        assert p["provenance"]["why"] == ""

    def test_approve_unknown_vocabulary_creates_enum_entry(self, tmp_path):
        """Approving a term for a completely new vocabulary creates the enum."""
        vm = self._make_vm(tmp_path)
        vm.add_term("novel_vocab", "novel_term")
        vm.approve_term("novel_vocab", "novel_term", approved_by="reviewer-1")
        assert "novel_term" in vm.get_enum("novel_vocab")
