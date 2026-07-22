"""bench-data-ingest workflow v0.15.0: test ingestion pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager

logger = logging.getLogger(__name__)


def run(
    fixture_dir: str,
    expected_min: int = 1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run bench-data-ingest against a fresh D-Base."""
    fixture_path = Path(fixture_dir)
    if not fixture_path.is_dir():
        raise ValueError(f"Fixture directory not found: {fixture_dir}")

    dbase = _create_fresh_dbase()
    try:
        record_counts, warnings = _extract_and_write(fixture_path, dbase)
        total = sum(record_counts.values())
        if total < expected_min:
            raise ValueError(
                f"Expected at least {expected_min} records, got {total}. Sources: {record_counts}"
            )
        report: dict[str, Any] = {
            "record_counts_by_source": record_counts,
            "total_records": total,
            "warnings": warnings,
        }
        if output_dir:
            import json

            out_path = Path(output_dir) / "ingest_report.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["output_path"] = str(out_path)
        return report
    finally:
        _discard_dbase(dbase)


def _create_fresh_dbase() -> DBase:
    tmpdir = tempfile.mkdtemp(prefix="dargus_ingest_test_")
    os.environ["DARGUS_HOME"] = tmpdir
    dbase_dir = Path(tmpdir) / "dbase"
    dbase_dir.mkdir(parents=True)
    return DBase("global", root_dir=Path(tmpdir))


def _discard_dbase(dbase: DBase) -> None:
    shutil.rmtree(dbase.dbase_dir.parent, ignore_errors=True)


def _extract_and_write(fixture_path: Path, dbase: DBase) -> tuple[dict[str, int], list[str]]:
    manager = DBaseManager(dbase)
    counts: dict[str, int] = {}
    warnings: list[str] = []

    for category, subdir in [("clinical", "clinical"), ("dti", "dti"), ("admet", "admet")]:
        cat_dir = fixture_path / subdir
        if not cat_dir.is_dir():
            continue
        n = 0
        for path in cat_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix in (".csv", ".tsv", ".tab"):
                try:
                    import pandas as pd

                    sep = "\t" if path.suffix in (".tsv", ".tab") else ","
                    df = pd.read_csv(path, sep=sep)
                    for _, row in df.iterrows():
                        try:
                            raw = row.to_dict()
                            record = manager.build_evidence(
                                {
                                    k: v
                                    for k, v in raw.items()
                                    if not (isinstance(v, float) and pd.isna(v))
                                },
                                source_metadata={
                                    "type": "file_path",
                                    "id": f"{subdir}/{path.name}",
                                },
                            )
                            manager.write_record(record)
                            n += 1
                        except Exception as exc:
                            warnings.append(f"{subdir}/{path.name}: {exc}")
                except Exception as exc:
                    warnings.append(f"{subdir}/{path.name} (parse): {exc}")
        counts[subdir] = n

    return counts, warnings
