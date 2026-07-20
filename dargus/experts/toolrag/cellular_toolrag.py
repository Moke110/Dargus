from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class CellularToolRAG(ToolRAG):
    """ToolRAG for cellular-level data: cell-based assays, proliferation, toxicity."""

    def __init__(self):
        super().__init__("cellular")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        from dargus.ingestion.converters.gdsc import GdscConverter

        self.registry.register(
            name="gdsc2_dose_response",
            template_id="cell_viability_assay_v1",
            match={
                "path_pattern": "GDSC2_fitted_dose_response*.csv",
                "columns_required": ["DRUG_ID", "CELL_LINE_NAME", "TCGA_DESC", "LN_IC50"],
            },
            field_mapping={
                "drug_id": "DRUG_ID",
                "cell_line_id": "CELL_LINE_NAME",
                "disease_id": "TCGA_DESC",
                "assay_type": "gdsc2_ln_ic50",
                "readout": "LN_IC50",
                "biological_level": "cellular",
            },
            biological_level="cellular",
            converter=GdscConverter,
        )
