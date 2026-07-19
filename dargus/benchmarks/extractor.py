"""Extract records from the global D-Base and build a stripped DBase-blank."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from dargus.dbase import DBase, TemplateRecord
from dargus.dbase.paths import default_dargus_home

logger = logging.getLogger(__name__)


class BenchmarkExtractor:
    """Extract matching records and create DBase-blank."""

    def __init__(self, work_dir: str | Path | None = None):
        self.work_dir = Path(work_dir or default_dargus_home() / "benchmark")
        self.blank_dir = self.work_dir / "blank"

    def extract(self, strip: dict[str, Any]) -> tuple[list[TemplateRecord], DBase]:
        """Return extracted records and a stripped blank D-Base."""
        records = self._query_global(strip)
        blank = self.create_blank(strip)
        return records, blank

    def create_blank(self, strip: dict[str, Any]) -> DBase:
        """Copy global D-Base to DBase-blank and remove matching records."""
        global_root = default_dargus_home()
        if self.blank_dir.exists():
            shutil.rmtree(self.blank_dir)
        self.blank_dir.mkdir(parents=True)
        shutil.copytree(global_root / "dbase", self.blank_dir / "dbase")
        blank = DBase("global", root_dir=self.blank_dir)
        to_remove = self._match_records(blank, strip)
        blank._records = [r for r in blank.list_records() if r not in to_remove]
        blank._record_ids = {r.record_id for r in blank._records}
        blank._manifest = [blank._record_to_manifest_entry(r) for r in blank._records]
        blank._dirty = True
        blank.save()
        return blank

    def _query_global(self, strip: dict[str, Any]) -> list[TemplateRecord]:
        dbase = DBase.global_instance()
        return self._match_records(dbase, strip)

    def _match_records(self, dbase: DBase, strip: dict[str, Any]) -> list[TemplateRecord]:
        records = dbase.list_records()
        for key, value in strip.items():
            records = [r for r in records if self._matches(r, key, value)]
        return records

    def _matches(self, record: TemplateRecord, key: str, value: Any) -> bool:
        if key == "source.type":
            return isinstance(record.source, dict) and record.source.get("type") == value
        return False
