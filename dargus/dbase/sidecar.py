"""D-Base v1.0.0 sidecars — append-only lifecycle/summary/embedding tables.

Three fields live outside the 50-field evidence record, each in its own
append-only sidecar file keyed by ``evidence_id`` (design/2.2):

- ``sidecars/status.jsonl``            — {evidence_id, status, superseded_by?}
- ``sidecars/llm_summary.jsonl``       — {evidence_id, summary}
- ``sidecars/embeddings-{model_fp}.jsonl`` — {evidence_id, vector}
- ``sidecars/embeddings_manifest.json`` — active + available model fingerprints

Sidecar entries never participate in the evidence_id identity hash.
Latest status entry per evidence_id wins; no entry means ``active``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

VALID_STATUSES = frozenset({"active", "superseded", "retracted", "holdout-test", "holdout-valid"})


def model_fingerprint(model_name: str) -> str:
    """Stable short fingerprint for an embedding model name."""
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]


class SidecarStore:
    """Append-only sidecar tables for one D-Base directory."""

    def __init__(self, dbase_dir: str | Path) -> None:
        self.sidecars_dir = Path(dbase_dir) / "sidecars"
        self.sidecars_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.sidecars_dir / "status.jsonl"
        self.summary_path = self.sidecars_dir / "llm_summary.jsonl"
        self.embeddings_manifest_path = self.sidecars_dir / "embeddings_manifest.json"

    # ── low-level append ────────────────────────────────────────────────────

    @staticmethod
    def _append(path: Path, entry: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _read(path: Path) -> Iterable[dict]:
        if not path.exists():
            return []
        entries = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    # ── status sidecar ──────────────────────────────────────────────────────

    def append_status(
        self, evidence_id: str, status: str, superseded_by: str | None = None
    ) -> None:
        """Append a lifecycle transition for an evidence record."""
        if status not in VALID_STATUSES:
            raise ValueError(f"status '{status}' not in {sorted(VALID_STATUSES)}")
        entry: dict[str, Any] = {"evidence_id": evidence_id, "status": status}
        if superseded_by is not None:
            entry["superseded_by"] = superseded_by
        self._append(self.status_path, entry)

    def read_status(self, evidence_id: str) -> dict:
        """Latest status for one record; no entry means active."""
        latest: dict = {"status": "active", "superseded_by": None}
        for entry in self._read(self.status_path):
            if entry.get("evidence_id") == evidence_id:
                latest = {"status": entry["status"], "superseded_by": entry.get("superseded_by")}
        return latest

    def read_all_status(self) -> dict[str, dict]:
        """Latest status per evidence_id across the sidecar."""
        latest: dict[str, dict] = {}
        for entry in self._read(self.status_path):
            eid = entry.get("evidence_id")
            if eid:
                latest[eid] = {
                    "status": entry["status"],
                    "superseded_by": entry.get("superseded_by"),
                }
        return latest

    # ── llm_summary sidecar ─────────────────────────────────────────────────

    def append_summary(self, evidence_id: str, summary: str) -> None:
        self._append(self.summary_path, {"evidence_id": evidence_id, "summary": summary})

    def read_summary(self, evidence_id: str) -> str | None:
        """Latest summary for one record, or None."""
        result: str | None = None
        for entry in self._read(self.summary_path):
            if entry.get("evidence_id") == evidence_id:
                result = entry.get("summary")
        return result

    # ── embeddings sidecars ─────────────────────────────────────────────────

    def embeddings_path(self, model_fp: str) -> Path:
        return self.sidecars_dir / f"embeddings-{model_fp}.jsonl"

    def append_embedding(self, evidence_id: str, vector: list[float], model_fp: str) -> None:
        self._append(self.embeddings_path(model_fp), {"evidence_id": evidence_id, "vector": vector})
        self._register_fingerprint(model_fp)

    def read_embeddings(self, model_fp: str) -> dict[str, list[float]]:
        """Latest vector per evidence_id for one model fingerprint."""
        vectors: dict[str, list[float]] = {}
        for entry in self._read(self.embeddings_path(model_fp)):
            eid = entry.get("evidence_id")
            if eid:
                vectors[eid] = entry.get("vector", [])
        return vectors

    # ── embeddings manifest ─────────────────────────────────────────────────

    def read_embeddings_manifest(self) -> dict:
        if self.embeddings_manifest_path.exists():
            return json.loads(self.embeddings_manifest_path.read_text(encoding="utf-8"))
        return {"active": None, "available": []}

    def _write_embeddings_manifest(self, manifest: dict) -> None:
        self.embeddings_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _register_fingerprint(self, model_fp: str) -> None:
        manifest = self.read_embeddings_manifest()
        if model_fp not in manifest["available"]:
            manifest["available"].append(model_fp)
        if manifest["active"] is None:
            manifest["active"] = model_fp
        self._write_embeddings_manifest(manifest)

    def set_active_fingerprint(self, model_fp: str) -> None:
        manifest = self.read_embeddings_manifest()
        if model_fp not in manifest["available"]:
            manifest["available"].append(model_fp)
        manifest["active"] = model_fp
        self._write_embeddings_manifest(manifest)

    def active_fingerprint(self) -> str | None:
        return self.read_embeddings_manifest()["active"]
