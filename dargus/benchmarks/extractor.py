"""Benchmark extractor v0.15.0 — evidence dict API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.dbase.paths import default_dargus_home

logger = logging.getLogger(__name__)


class BenchmarkExtractor:
    """Extract matching evidence records and create DBase-blank."""

    def __init__(self, work_dir: str | Path | None = None):
        self.work_dir = Path(work_dir or default_dargus_home() / "benchmark")
        self.blank_dir = self.work_dir / "blank"

    def extract(self, strip: dict[str, Any]) -> tuple[list[dict], DBase]:
        """Return extracted records and a stripped blank D-Base."""
        records = self._query_global(strip)
        blank = self.create_blank(strip)
        return records, blank

    def create_blank(self, strip: dict[str, Any]) -> DBase:
        """Create blank D-Base by stripping matching evidence_ids."""
        blank = DBase("global", root_dir=self.blank_dir)
        match_ids = {
            r.get("evidence_id") for r in self._query_global(strip) if r.get("evidence_id")
        }
        all_records = DBase.global_instance().read_shards()
        kept = [r for r in all_records if r.get("evidence_id") not in match_ids]
        for r in kept:
            blank.append_shard(r)
        blank.mark_view_stale()
        return blank

    def _query_global(self, strip: dict[str, Any]) -> list[dict]:
        dbase = DBase.global_instance()
        records = dbase.read_shards()
        for key, value in strip.items():
            records = [r for r in records if self._matches(r, key, value)]
        return records

    def _matches(self, record: dict, key: str, value: Any) -> bool:
        if key == "source.type":
            sources = record.get("sources", [])
            return any(s.get("type") == value for s in sources)
        return str(record.get(key, "")) == str(value)
