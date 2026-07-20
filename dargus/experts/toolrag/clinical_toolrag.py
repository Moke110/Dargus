from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class ClinicalToolRAG(ToolRAG):
    """ToolRAG for clinical-level data: trial results, patient cohorts."""

    def __init__(self):
        super().__init__("clinical")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        from dargus.ingestion.converters.top_clinical import TopClinicalConverter

        # Register specific patterns first
        self.registry.register(
            name="raw_data_clinical",
            template_id="clinical_trial_outcome_v1",
            match={
                "path_pattern": "raw_data*.csv",
                "columns_required": ["diseases", "drugs", "label", "phase"],
            },
            field_mapping={
                "drug_id": "drugs",
                "disease_id": "diseases",
                "endpoint": "trial_success",
                "fold_change": "label",
                "phase": "phase",
                "biological_level": "clinical",
            },
            biological_level="clinical",
            converter=TopClinicalConverter,
        )

        self.registry.register(
            name="phase_clinical_trials",
            template_id="clinical_trial_outcome_v1",
            match={
                "path_pattern": "phase_*.csv",
                "columns_required": ["diseases", "drugs", "label", "phase"],
            },
            field_mapping={
                "drug_id": "drugs",
                "disease_id": "diseases",
                "endpoint": "trial_success",
                "fold_change": "label",
                "phase": "phase",
                "biological_level": "clinical",
            },
            biological_level="clinical",
            converter=TopClinicalConverter,
        )

        # Generic catch-all last
        self.registry.register(
            name="top_clinical_trials",
            template_id="clinical_trial_outcome_v1",
            match={
                "path_pattern": "*.csv",
                "columns_required": ["diseases", "drugs", "label", "phase"],
            },
            field_mapping={
                "drug_id": "drugs",
                "disease_id": "diseases",
                "endpoint": "trial_success",
                "fold_change": "label",
                "phase": "phase",
                "biological_level": "clinical",
            },
            biological_level="clinical",
            converter=TopClinicalConverter,
        )
