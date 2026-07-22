"""Train workflow v0.15.0 — ingest data into the global D-Base."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.experts.types import IngestionSummary

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Report from a Train workflow run."""

    n_records: int = 0
    n_skipped: int = 0
    dbase_size: int = 0
    errors: list[str] = field(default_factory=list)


def run(datadir: str, reset: bool = False, disease_kb_dir: str | None = None) -> TrainingReport:
    """Ingest data files into the global D-Base and return a TrainingReport."""
    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)

    if reset:
        manager.reset()

    n_records = 0
    errors: list[str] = []

    datadir_path = Path(datadir)
    if datadir_path.is_dir():
        for yaml_path in sorted(datadir_path.glob("*.yaml")):
            try:
                import yaml

                with yaml_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                if isinstance(data, dict):
                    record = manager.build_evidence(
                        data,
                        source_metadata={
                            "type": "db_accession",
                            "id": f"file:{yaml_path.name}",
                        },
                    )
                    manager.write_record(record)
                    n_records += 1
            except Exception as exc:
                errors.append(f"{yaml_path.name}: {exc}")

    records = dbase.read_shards()
    return TrainingReport(
        n_records=n_records,
        n_skipped=0,
        dbase_size=len(records),
        errors=errors,
    )


def ingest_report(datadir: str, disease_kb_dir: str | None = None) -> IngestionSummary:
    """Generate an ingestion report without writing to D-Base."""
    dbase = DBase.global_instance()
    manager = DBaseManager(dbase)
    records_before = len(dbase.read_shards())
    # Dry-run: just count what would be ingested
    datadir_path = Path(datadir)
    n_files = len(list(datadir_path.glob("*.yaml"))) if datadir_path.is_dir() else 0
    return IngestionSummary(
        total_instances=n_files,
        per_level={},
        duplicates=0,
        errors=0,
        total_instances_before=records_before,
        total_instances_after=records_before,
    )
