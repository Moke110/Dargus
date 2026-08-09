"""D-Base vocabulary manager — three-axis enum registry loaded from vocabularies.json.

Human-approval-gated vocabulary extension (design principle 4):
  - add_term(vocabulary, term, provenance)  → stages as pending
  - approve_term(vocabulary, term, approved_by) → activates
  - reject_term(vocabulary, term) → discards
  - pending_terms() → dict of pending terms

Pending terms are persisted in a sidecar file (vocabularies_pending.json) and
survive manager restarts. The validator accepts only active terms — pending
terms fail validation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── pending-state persistence ─────────────────────────────────────────────────

_DEFAULT_PENDING_PATH = Path(__file__).resolve().parent / "vocabularies_pending.json"


class VocabularyManager:
    """CURIE prefix registry + three-axis enum term registry.

    Loads all controlled vocabularies from vocabularies.json (§3.0–§3.14).

    Gated term extension — design principle 4:
        add_term(vocabulary, term)         → stage as pending (needs approval)
        approve_term(vocabulary, term, approved_by) → activate
        reject_term(vocabulary, term)      → discard pending
    """

    def __init__(self, pending_path: str | Path | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._curie_patterns: dict[str, re.Pattern] = {}
        self._pending: dict[str, list[dict]] = {}
        self._pending_path: str = str(pending_path or _DEFAULT_PENDING_PATH)
        self._load()

    def _load(self) -> None:
        path = Path(__file__).resolve().parent / "vocabularies.json"
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

        # compile CURIE patterns
        curie_data = self._data.get("curie_prefixes") or {}
        for prefix, pat in (curie_data.get("hard_validated") or {}).items():
            self._curie_patterns[prefix] = re.compile(pat)

        # load pending terms sidecar
        self._load_pending()

    # ── pending state persistence ─────────────────────────────────────────────

    def _load_pending(self) -> None:
        ppath = Path(self._pending_path)
        if ppath.exists():
            try:
                self._pending = json.loads(ppath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._pending = {}

    def _save_pending(self) -> None:
        Path(self._pending_path).write_text(
            json.dumps(self._pending, indent=2, sort_keys=True), encoding="utf-8"
        )

    # ── vocabularies.json saving (public) ──────────────────────────────────────

    def get_enum_values(self, vocab_name: str) -> list[str]:
        """Return flat list of enum values for a vocabulary."""
        entry = self._data.get(vocab_name) or {}
        if isinstance(entry, dict):
            vals = entry.get("values", [])
            if vals and isinstance(vals[0], dict):
                return [item["value"] for item in vals]
            return list(vals)
        return []

    def get_enum(self, vocab_name: str) -> list[str]:
        """Alias for get_enum_values."""
        return self.get_enum_values(vocab_name)

    def is_valid_enum_value(self, vocab_name: str, value: str) -> bool:
        return value in self.get_enum_values(vocab_name)

    def clinical_levels(self) -> frozenset:
        entry = self._data.get("biological_level") or {}
        vals = entry.get("values", [])
        return frozenset(item["value"] for item in vals if item.get("is_clinical"))

    def sim_levels(self) -> frozenset:
        entry = self._data.get("biological_level") or {}
        vals = entry.get("values", [])
        return frozenset(item["value"] for item in vals if item.get("is_sim"))

    def log_effect_types(self) -> frozenset:
        entry = self._data.get("y_effect_value_type") or {}
        return frozenset(entry.get("log_types", []))

    def control_labels(self) -> frozenset:
        return frozenset(self.get_enum_values("x_value_control_labels"))

    def validate_curie(self, curie_str: str) -> bool:
        """Check if a CURIE string has a registered prefix and valid accession."""
        if ":" not in curie_str:
            return False
        prefix, _, accession = curie_str.partition(":")

        curie_data = self._data.get("curie_prefixes") or {}
        fallback = set(curie_data.get("fallback", []))

        if prefix in fallback:
            return True
        pat = self._curie_patterns.get(prefix)
        if pat is None:
            return False
        return bool(pat.match(accession))

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyManager":
        vm = cls()
        if Path(path).exists():
            vm._data = json.loads(Path(path).read_text(encoding="utf-8"))
        return vm

    # ── full-state save/load (vocabularies.json + pending) ─────────────────────

    def save_state(self, path: str | Path) -> None:
        """Persist vocab data + pending terms.

        If *path* is a directory (or has no file extension), writes
        ``vocabularies.json`` and ``vocabularies_pending.json`` inside it.
        If *path* is a file, writes vocab data there and pending terms to
        a sibling ``<stem>_pending.json`` file.
        """
        dest = Path(path)
        if dest.is_dir() or not dest.suffix:
            # directory: write both files inside
            dest.mkdir(parents=True, exist_ok=True)
            self.save(dest / "vocabularies.json")
            self._pending_path = str(dest / "vocabularies_pending.json")
            self._save_pending()
        else:
            # file: write vocab data to path, pending alongside
            self.save(dest)
            pending_sibling = dest.with_suffix("").with_suffix("")
            pending_file = dest.parent / f"{pending_sibling.name}_pending.json"
            # avoid a plain "_pending.json" when path has no stem
            if not pending_sibling.name:
                pending_file = dest.parent / "vocabularies_pending.json"
            self._pending_path = str(pending_file)
            self._save_pending()

    @classmethod
    def load_state(cls, path: str | Path) -> "VocabularyManager":
        """Load from a state directory or file.

        If *path* is a directory, loads ``vocabularies.json`` and
        ``vocabularies_pending.json`` from it. If *path* is a file, loads
        that file as the vocab data and looks for a sibling
        ``<stem>_pending.json`` for pending terms.
        """
        dest = Path(path)
        if dest.is_dir():
            data_path = dest / "vocabularies.json"
            pending_path = dest / "vocabularies_pending.json"
        elif dest.suffix:
            # file: pending is sibling <stem>_pending.json
            stem = dest.with_suffix("").name
            pending_path = dest.parent / f"{stem}_pending.json"
            data_path = dest
        else:
            # no extension — treat as directory-like path
            pending_path = dest.parent / "vocabularies_pending.json"
            data_path = dest

        vm = cls(pending_path=pending_path)
        if data_path.exists():
            vm._data = json.loads(data_path.read_text(encoding="utf-8"))
        vm._load_pending()
        return vm

    # ── gated add_term API ─────────────────────────────────────────────────────

    def pending_terms(self) -> dict[str, list[dict]]:
        """Return a copy of all pending terms, keyed by vocabulary name."""
        return {k: list(v) for k, v in self._pending.items()}

    def add_term(
        self,
        vocabulary: str,
        term: str,
        provenance: dict[str, str] | None = None,
    ) -> None:
        """Stage a new term as pending (requires human approval to activate).

        Args:
            vocabulary: The vocabulary name (e.g. ``"evidence_design"``).
            term: The term to add.
            provenance: Optional dict with who/why keys. Defaults to
                        ``{"who": "unknown", "why": ""}``. A ``when``
                        ISO-8601 timestamp is always recorded.
        """
        # never stage if already in the active set
        if term in self.get_enum_values(vocabulary):
            return

        prov = dict(provenance or {})
        prov.setdefault("who", "unknown")
        prov.setdefault("why", "")
        prov["when"] = datetime.now(timezone.utc).isoformat()

        entry = {"term": term, "provenance": prov}

        if vocabulary not in self._pending:
            self._pending[vocabulary] = []

        # deduplicate by term
        existing_idx = None
        for i, p in enumerate(self._pending[vocabulary]):
            if p.get("term") == term:
                existing_idx = i
                break
        if existing_idx is not None:
            self._pending[vocabulary][existing_idx] = entry
        else:
            self._pending[vocabulary].append(entry)

        self._save_pending()

    def approve_term(self, vocabulary: str, term: str, approved_by: str) -> None:
        """Approve a pending term, moving it into the active vocabulary.

        Args:
            vocabulary: The vocabulary name.
            term: The term to approve (must be in pending).
            approved_by: Non-empty string identifying who approved.

        Raises:
            ValueError: If *approved_by* is empty or the term is not in pending.
        """
        if not approved_by or not approved_by.strip():
            raise ValueError(
                "approved_by must be a non-empty string identifying the human approver"
            )

        # find the pending entry
        pending_list = self._pending.get(vocabulary, [])
        idx = None
        for i, p in enumerate(pending_list):
            if p.get("term") == term:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Term '{term}' not in pending for vocabulary '{vocabulary}'")

        # remove from pending
        del self._pending[vocabulary][idx]
        if not self._pending[vocabulary]:
            del self._pending[vocabulary]

        # add to active set
        self._activate_term(vocabulary, term)

        self._save_pending()

    def reject_term(self, vocabulary: str, term: str) -> None:
        """Discard a pending term without activating it.

        No-op if the term is not in pending.
        """
        pending_list = self._pending.get(vocabulary, [])
        idx = None
        for i, p in enumerate(pending_list):
            if p.get("term") == term:
                idx = i
                break
        if idx is not None:
            del self._pending[vocabulary][idx]
            if not self._pending[vocabulary]:
                del self._pending[vocabulary]
            self._save_pending()

    def _activate_term(self, vocabulary: str, term: str) -> None:
        """Move a term into the active vocabulary data structure."""
        entry = self._data.get(vocabulary)
        if entry is None:
            # new vocabulary: create as list-style
            self._data[vocabulary] = {"description": "", "values": [term]}
        elif isinstance(entry, dict):
            vals = entry.get("values", [])
            if vals and isinstance(vals[0], dict):
                # object-style values (like biological_level)
                vals.append({"value": term})
            else:
                vals.append(term)
            entry["values"] = vals
        elif isinstance(entry, list):
            entry.append(term)
        else:
            self._data[vocabulary] = {"description": "", "values": [term]}
