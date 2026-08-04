"""Ingest converter framework — canonical conversion path.

A converter is a ``BaseConverter`` subclass that turns one source's raw
records (``raw.jsonl`` provenance wrappers) into three-axis evidence dicts,
reporting structured skip reasons. ``convert_slice`` drives a converter over
a slice's raw directory and writes:

  * per-biological-level evidence JSONL under ``<out>/<level>.jsonl``
  * a complete per-record skip manifest (``skips.jsonl``) — every skipped
    record with a specific reason, no truncation.

The framework keeps the no-sidecar-fields contract: evidence records carry
only the validated 50-field shape (plus ``evidence_id``/``is_clinical`` added
by ``build_evidence``/``validate_evidence``).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.dbase.store import DBaseStore
from dargus.ingestion.converters.base import BaseConverter, SkipRecord

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Outcome of converting one source's raw records."""

    records: list[dict[str, Any]] = field(default_factory=list)
    skips: list[SkipRecord] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def skip(self, source_entry: str, source: str, reason: str, detail: str = "") -> None:
        self.skips.append(
            SkipRecord(
                source_entry=source_entry,
                source=source,
                reason=reason,
                detail=detail,
            )
        )

    @property
    def n_skipped(self) -> int:
        return len(self.skips)

    @property
    def n_records(self) -> int:
        return len(self.records)

    def skip_counter(self) -> Counter:
        return Counter(s.reason for s in self.skips)


def write_manifest(skips: list[SkipRecord], path: Path) -> None:
    """Write the complete per-record skip manifest as JSONL (no truncation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in skips:
            fh.write(
                json.dumps(
                    {
                        "source": s.source,
                        "source_entry": s.source_entry,
                        "reason": s.reason,
                        "detail": s.detail,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def convert_slice(
    converter: BaseConverter,
    raw_dir: Path,
    out_dir: Path,
    project_id: str,
    by_level: bool = True,
) -> dict[str, Any]:
    """Convert every raw wrapper in *raw_dir* through *converter*.

    Each converter output is driven through ``DBaseStore.build_evidence`` (the
    canonical seam: it assembles the three-axis dict, injects provenance,
    validates against the hard rules, and content-addresses the record).
    Validation failures become ``validation:<errors>`` skips. Evidence is
    written as JSONL grouped by ``biological_level`` into *out_dir*, plus a
    complete ``skips.jsonl`` manifest (every skipped record, no truncation).

    A fresh in-memory D-Base under ``out_dir`` is used so conversion is
    self-contained and never touches a project store.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dbase = DBase(project_id, root_dir=Path(tmp))
        manager = DBaseStore(dbase)
        result = ConversionResult()
        for raw in _iter_raw(raw_dir):
            source_entry = str(raw.get("source_entry", ""))
            source = str(raw.get("source", ""))
            for item in converter.convert(raw):
                if isinstance(item, SkipRecord):
                    result.skips.append(item)
                    continue
                try:
                    record = manager.build_evidence(
                        item,
                        source_metadata={
                            "type": "database",
                            "name": converter.template_id,
                            "entry": source_entry,
                            "time": str(raw.get("source_time", "")),
                        },
                    )
                except ValueError as exc:
                    result.skip(
                        source_entry=source_entry,
                        source=source or converter.template_id,
                        reason="validation",
                        detail=str(exc),
                    )
                    continue
                result.add(record)

    # dedup by evidence_id — duplicate raw rows (e.g. the same NCT appearing
    # several times in the slice) collapse to one content-addressed record,
    # exactly as ``DBaseStore.write_record`` would store them.
    seen_ids: set[str] = set()
    unique_records: list[dict] = []
    n_deduped = 0
    for rec in result.records:
        eid = rec.get("evidence_id") or ""
        if eid in seen_ids:
            n_deduped += 1
            continue
        seen_ids.add(eid)
        unique_records.append(rec)
    result.records = unique_records

    out_dir.mkdir(parents=True, exist_ok=True)
    by_level_files: dict[str, Path] = {}
    if by_level:
        groups: dict[str, list[dict]] = {}
        for rec in result.records:
            groups.setdefault(rec.get("biological_level", "other"), []).append(rec)
        for level, recs in groups.items():
            path = out_dir / f"{level}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            by_level_files[level] = path
    else:
        path = out_dir / "evidence.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in result.records:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        by_level_files["all"] = path

    write_manifest(result.skips, out_dir / "skips.jsonl")
    summary = {
        "converter": type(converter).__name__,
        "n_records": result.n_records,
        "n_skipped": result.n_skipped,
        "n_deduped": n_deduped,
        "by_level": {level: len(recs) for level, recs in groups.items()} if by_level else {},
        "skip_reasons": dict(result.skip_counter()),
    }
    # A tiny projection of the stored records stays invisible here; the files
    # are the authoritative output. Persist the summary alongside.
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _iter_raw(raw_dir: Path):
    raw_file = raw_dir / "raw.jsonl"
    if raw_file.exists():
        for line in raw_file.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed raw line in %s", raw_file)
                continue
        return
    # fallback: glob *.jsonl directly under raw_dir
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed raw line in %s", path)
                continue
