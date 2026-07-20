"""Train workflow — ingest data into the global D-Base."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dargus.dbase import DBase
from dargus.dbase.manager import DBaseManager
from dargus.experts.disease import DiseaseExpert
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
    _ensure_default_templates(dbase)
    manager = DBaseManager(dbase)
    disease_expert = DiseaseExpert(manager)

    if reset:
        manager.reset()

    summary = disease_expert.ingest_from_dir(
        datadir, disease_kb_dir=disease_kb_dir, auto_confirm=True
    )

    return TrainingReport(
        n_records=summary.total_instances,
        n_skipped=0,
        dbase_size=len(dbase.list_records()),
        errors=[],
    )


def ingest_report(datadir: str, disease_kb_dir: str | None = None) -> IngestionSummary:
    """Generate an ingestion report without writing to D-Base."""
    dbase = DBase.global_instance()
    _ensure_default_templates(dbase)
    manager = DBaseManager(dbase)
    expert = DiseaseExpert(manager)
    return expert.ingest_from_dir(datadir, disease_kb_dir=disease_kb_dir, auto_confirm=False)


def _ensure_default_templates(dbase: DBase) -> None:
    drug_vocab = "global_drug_vocab"
    disease_vocab = "global_disease_vocab"
    endpoint_vocab = "global_endpoint_vocab"
    if "clinical_trial_outcome_v1" not in dbase._templates:
        from dargus.dbase import TemplateSchema

        dbase.add_template(
            TemplateSchema(
                template_id="clinical_trial_outcome_v1",
                fields=[
                    {
                        "name": "biological_level",
                        "type": "factor",
                        "vocabulary": [
                            "molecular",
                            "cellular",
                            "exvivo",
                            "animal",
                            "clinical",
                            "epi",
                        ],
                    },
                    {"name": "drug_id", "type": "factor", "vocabulary_ref": drug_vocab},
                    {"name": "disease_id", "type": "factor", "vocabulary_ref": disease_vocab},
                    {"name": "endpoint", "type": "factor", "vocabulary_ref": endpoint_vocab},
                    {"name": "fold_change", "type": "float"},
                    {"name": "ci95_lower", "type": "float"},
                    {"name": "ci95_upper", "type": "float"},
                ],
            )
        )
