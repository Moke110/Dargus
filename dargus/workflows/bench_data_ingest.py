"""bench-data-ingest workflow: test ingestion pipeline against a fresh empty D-Base."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.experts.levels import ClinicalExpert, MolecularExpert
from dargus.ingestion.converters.tdc_admet import TdcAdmetConverter
from dargus.workflows.train import _ensure_default_templates

logger = logging.getLogger(__name__)


def run(
    fixture_dir: str,
    expected_min: int = 1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run bench-data-ingest: extract, convert, write to a fresh D-Base, verify, discard.

    Args:
        fixture_dir: Path containing ``dti/``, ``clinical/``, ``admet/`` subdirectories.
        expected_min: Minimum total records expected (default 1).
        output_dir: Optional directory for the summary JSON report.

    Returns:
        Dict with keys ``record_counts_by_source``, ``total_records``, ``warnings``.
    """
    fixture_path = Path(fixture_dir)
    if not fixture_path.is_dir():
        raise ValueError(f"Fixture directory not found: {fixture_dir}")

    dbase = _create_fresh_dbase()

    try:
        record_counts, warnings = _extract_and_write(fixture_path, dbase)

        total = sum(record_counts.values())
        if total < expected_min:
            raise ValueError(
                f"Expected at least {expected_min} records, got {total}. "
                f"Sources: {record_counts}"
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

        logger.info(
            "bench-data-ingest complete: %d records from %d sources",
            total,
            len(record_counts),
        )
        return report

    finally:
        _discard_dbase(dbase)


def _create_fresh_dbase() -> DBase:
    """Create a new empty D-Base in a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="dargus_ingest_test_")
    os.environ["DARGUS_HOME"] = tmpdir
    dbase_dir = Path(tmpdir) / "dbase"
    dbase_dir.mkdir(parents=True)
    (dbase_dir / "templates").mkdir()
    dbase = DBase("global", root_dir=Path(tmpdir))
    _ensure_default_templates(dbase)
    return dbase


def _discard_dbase(dbase: DBase) -> None:
    """Remove the temp D-Base directory."""
    root = dbase.dbase_dir.parent
    shutil.rmtree(root, ignore_errors=True)


def _extract_and_write(fixture_path: Path, dbase: DBase) -> tuple[dict[str, int], list[str]]:
    """Scan fixture subdirectories, extract + write records, return counts and warnings."""
    manager = DBaseManager(dbase)
    counts: dict[str, int] = {}
    warnings: list[str] = []

    # clinical/ — use ClinicalExpert.extract()
    clinical_dir = fixture_path / "clinical"
    if clinical_dir.is_dir():
        expert = ClinicalExpert(dbase=dbase)
        report = expert.extract(str(clinical_dir))
        for inst in report.instances:
            try:
                record = manager.fill_template(
                    inst.raw_fields,
                    source_metadata={
                        "type": "bench_ingest_test",
                        "source": inst.source_file or "clinical_fixture",
                    },
                    suggested_template=inst.template_id,
                )
                manager.write_record(record)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"clinical/{inst.source_file}: {exc}")
        counts["clinical"] = report.n_instances

    # dti/ — use MolecularExpert.extract()
    dti_dir = fixture_path / "dti"
    if dti_dir.is_dir():
        expert = MolecularExpert(dbase=dbase)
        report = expert.extract(str(dti_dir))
        for inst in report.instances:
            try:
                record = manager.fill_template(
                    inst.raw_fields,
                    source_metadata={
                        "type": "bench_ingest_test",
                        "source": inst.source_file or "dti_fixture",
                    },
                    suggested_template=inst.template_id,
                )
                manager.write_record(record)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"dti/{inst.source_file}: {exc}")
        counts["dti"] = report.n_instances

    # admet/ — use TdcAdmetConverter
    admet_dir = fixture_path / "admet"
    if admet_dir.is_dir():
        n = 0
        for path in admet_dir.iterdir():
            if not path.is_file():
                continue
            for assay in [
                "solubility",
                "lipophilicity",
                "caco2",
                "ppb",
                "ppbr",
                "ld50",
                "bioavailability",
            ]:
                try:
                    converter = TdcAdmetConverter(assay_name=assay)
                    rows = converter.convert(path)
                    for row in rows:
                        try:
                            record = manager.fill_template(
                                row,
                                source_metadata={
                                    "type": "bench_ingest_test",
                                    "source": f"admet/{path.name}",
                                },
                                suggested_template=converter.template_id,
                            )
                            manager.write_record(record)
                            n += 1
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(f"admet/{path.name}: {exc}")
                    break
                except Exception:
                    continue
        counts["admet"] = n

    dbase.save()
    return counts, warnings
