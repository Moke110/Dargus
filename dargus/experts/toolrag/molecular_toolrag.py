from __future__ import annotations

from dargus.experts.toolrag.base import ToolRAG
from dargus.experts.toolrag.registry import ConverterRegistry


class MolecularToolRAG(ToolRAG):
    """ToolRAG for molecular-level data: DTI, ADMET, binding assays."""

    def __init__(self):
        super().__init__("molecular")
        self.registry = ConverterRegistry()
        self._load_registry()

    def _load_registry(self) -> None:
        self.registry.register(
            name="tdc_dti_bindingdb",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "bindingdb_*.csv",
                "columns_required": ["ID1", "ID2", "Y"],
            },
            field_mapping={
                "drug_id": "ID1",
                "target_id": "ID2",
                "readout": "Y",
                "assay_type": "binding_affinity",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )
