"""Standalone smoke: D-Base — real validate + write/read round-trip (offline).

Pins the D-Base behavioral invariant: a valid Evidence Record is validated,
written through the real store, read back, and validated again — deterministically,
without any network or embedding model.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_dbase.py
"""

from __future__ import annotations

import os
import sys
import tempfile

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


def _make_evidence(**overrides: object) -> dict:
    """A valid v1.0.0 three-axis evidence dict (descriptive, xy.count=1)."""
    e = {
        "biological_level": "molecular",
        "evidence_design": "descriptive",
        "xy": {"count": 1},
        "x": {
            "type": "drug",
            "value": [{"entity_id": "chembl:CHEMBL25", "entity_label": "aspirin"}],
        },
        "y": {
            "type": "logP",
            "category": "pk_adme",
            "value": [3.5],
            "assay": "binding_assay",
        },
        "bg": {"disease_id": [], "drugs": [], "genes": []},
        "sources": [{"rank": 1, "type": "journal", "name": "10.1234/test"}],
        "source_entry": "10.1234/test",
        "source_time": "2026-01-01",
    }
    e.update(overrides)
    return e


class _OfflineEmbeddingBackend:
    """Deterministic offline stand-in for the default embedding backend.

    The D-Base write path embeds into a sidecar (best-effort); the real
    SentenceTransformer backend downloads a model from Hugging Face on first
    embed, which would hang offline. Injecting this stub keeps the smoke
    deterministic and offline (conftest.py prior art).
    """

    _model_name = "smoke-offline"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 384
            for i, b in enumerate(text.encode("utf-8")):
                vec[i % 384] += float(b)
            norm = sum(v * v for v in vec) ** 0.5
            vectors.append([v / norm for v in vec] if norm else vec)
        return vectors


def main() -> int:
    # Run against an isolated temp D-Base, never the real ~/.dargus.
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DARGUS_HOME"] = tmp

        from dargus.dbase import DBase
        from dargus.dbase.store import DBaseStore
        from dargus.dbase.validate import validate_evidence
        from dargus.models.embedding import EmbeddingModel

        dbase = DBase("global", root_dir=tmp)
        store = DBaseStore(dbase, embedding_model=EmbeddingModel(_OfflineEmbeddingBackend()))

        # 1. A valid record passes validation.
        evidence = _make_evidence()
        result = validate_evidence(evidence)
        assert result.ok, f"validation failed: {result.hard_errors}"

        # 2. Write through the real store; first write returns True.
        ok = store.write_record(evidence)
        assert ok is True, f"write_record returned {ok!r}"

        # 3. Read it back from the real shard store.
        records = dbase.read_shards()
        assert len(records) == 1, f"expected 1 shard record, got {len(records)}"
        record = records[0]
        evidence_id = record.get("evidence_id", "")
        assert evidence_id.startswith("ev_"), f"bad evidence_id {evidence_id!r}"
        assert record["x"]["value"][0]["entity_id"] == "chembl:CHEMBL25"

        # 4. Read back through the store's read_record and re-validate.
        loaded = store.read_record(evidence_id)
        assert loaded is not None, "read_record returned None"
        reloaded = validate_evidence(loaded)
        assert reloaded.ok, f"re-read record failed validation: {reloaded.hard_errors}"

        # 5. A genuinely invalid record is rejected by the real validator.
        bad = _make_evidence(biological_level="not_a_level")
        assert not validate_evidence(bad).ok, "invalid record passed validation"

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
