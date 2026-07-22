"""D-Base v0.15.0 — keyed-object evidence store with shard JSONL + Parquet view."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from pathlib import Path

from dargus.dbase.paths import global_dbase_root
from dargus.dbase.vocabulary import VocabularyManager


class DBase:
    """Evidence store: shard JSONL (authoritative) + Parquet view (derived)."""

    def __init__(self, project_id: str, root_dir: str | Path):
        self.project_id = project_id
        self.root = Path(root_dir)
        self.dbase_dir = self.root / "dbase"
        self.data_dir = self.dbase_dir / "data"
        self.views_dir = self.dbase_dir / "views"
        self.vocab_path = self.dbase_dir / "vocabularies.json"
        self.manifest_path = self.dbase_dir / "manifest.json"
        self.parquet_path = self.views_dir / "dbase.parquet"
        self.quarantine_path = self.dbase_dir / "migration_quarantine.jsonl"

        self._vocab: VocabularyManager | None = None
        self._writer_id: str = uuid.uuid4().hex[:8]

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.views_dir.mkdir(parents=True, exist_ok=True)

    @property
    def vocab(self) -> VocabularyManager:
        if self._vocab is None:
            if self.vocab_path.exists():
                self._vocab = VocabularyManager.load(self.vocab_path)
            else:
                self._vocab = VocabularyManager()
        return self._vocab

    @property
    def shard_path(self) -> Path:
        return self.data_dir / f"shard-{self._writer_id}.jsonl"

    @property
    def field_registry_path(self) -> Path:
        return self.dbase_dir / "field_registry.yaml"

    # ── shard I/O ─────────────────────────────────────────────────────────

    def read_shards(self) -> list[dict]:
        """Read all evidence records from all shard JSONL files."""
        records: list[dict] = []
        for shard in sorted(self.data_dir.glob("shard-*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def read_shard_ids(self) -> set[str]:
        """Read all evidence_ids from all shards (for dedup)."""
        ids: set[str] = set()
        for shard in sorted(self.data_dir.glob("shard-*.jsonl")):
            with shard.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        eid = rec.get("evidence_id")
                        if eid:
                            ids.add(eid)
                    except json.JSONDecodeError:
                        continue
        return ids

    def append_shard(self, record: dict) -> None:
        """Append one evidence record to the writer's shard with flock."""
        record.setdefault("evidence_id", "")
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.shard_path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def evidence_id_exists(self, evidence_id: str) -> bool:
        """Check if an evidence_id already exists in any shard."""
        return evidence_id in self.read_shard_ids()

    # ── manifest ──────────────────────────────────────────────────────────

    def read_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"shards": {}, "view_built_at": None, "row_count": 0}

    def write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def mark_view_stale(self) -> None:
        manifest = self.read_manifest()
        # update row count from all shards
        count = 0
        shard_info = {}
        for shard in sorted(self.data_dir.glob("shard-*.jsonl")):
            n = sum(1 for _ in shard.open("r", encoding="utf-8"))
            shard_info[shard.name] = n
            count += n
        manifest["shards"] = shard_info
        manifest["row_count"] = count
        manifest["view_built_at"] = None
        self.write_manifest(manifest)

    def rebuild_view(self) -> None:
        """Rebuild views/dbase.parquet from all shard JSONL."""
        try:
            import pandas as pd
        except ImportError:
            return  # parquet optional; skip if pandas not installed

        records = self.read_shards()
        if not records:
            return

        df = pd.DataFrame(records)
        self.views_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, index=False)

        manifest = self.read_manifest()
        manifest["view_built_at"] = str(pd.Timestamp.now())
        self.write_manifest(manifest)

    def query_parquet(
        self,
        readout_type: str | None = None,
        readout_category: str | None = None,
        intervention_id: str | None = None,
        disease_id: str | None = None,
        biological_level: str | None = None,
    ) -> list[dict]:
        """Query via Parquet view. Falls back to shard scan if parquet unavailable."""
        try:
            import pandas as pd
        except ImportError:
            return self._query_shards(
                readout_type,
                readout_category,
                intervention_id,
                disease_id,
                biological_level,
            )

        if not self.parquet_path.exists():
            self.rebuild_view()

        if not self.parquet_path.exists():
            return []

        df = pd.read_parquet(self.parquet_path)
        if readout_type:
            df = df[df["readout_type"] == readout_type]
        if readout_category:
            df = df[df["readout_category"] == readout_category]
        if intervention_id:
            mask = df["interventions"].apply(
                lambda ivs: (
                    any(i.get("entity_id") == intervention_id for i in (ivs or []))
                    if isinstance(ivs, list)
                    else False
                )
            )
            df = df[mask]
        if disease_id:
            df = df[df["disease_id"] == disease_id]
        if biological_level:
            df = df[df["biological_level"] == biological_level]
        return df.to_dict(orient="records")

    def _query_shards(
        self,
        readout_type: str | None = None,
        readout_category: str | None = None,
        intervention_id: str | None = None,
        disease_id: str | None = None,
        biological_level: str | None = None,
    ) -> list[dict]:
        results = self.read_shards()
        if readout_type:
            results = [r for r in results if r.get("readout_type") == readout_type]
        if readout_category:
            results = [r for r in results if r.get("readout_category") == readout_category]
        if intervention_id:
            results = [
                r
                for r in results
                if any(
                    i.get("entity_id") == intervention_id for i in (r.get("interventions") or [])
                )
            ]
        if disease_id:
            results = [r for r in results if r.get("disease_id") == disease_id]
        if biological_level:
            results = [r for r in results if r.get("biological_level") == biological_level]
        return results

    def clear(self) -> None:
        """Remove all shards, views, and manifest."""
        for shard in self.data_dir.glob("shard-*.jsonl"):
            shard.unlink()
        if self.parquet_path.exists():
            self.parquet_path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    @classmethod
    def global_instance(cls) -> "DBase":
        return cls(project_id="global", root_dir=global_dbase_root())
