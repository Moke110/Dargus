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
        # BindingDB variants
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

        self.registry.register(
            name="tdc_dti_bindingdb_ki",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "bindingdb_ki*.csv",
                "columns_required": ["Drug", "Target", "Y"],
            },
            field_mapping={
                "drug_id": "Drug",
                "target_id": "Target",
                "readout": "Y",
                "assay_type": "binding_affinity_ki",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )

        self.registry.register(
            name="tdc_dti_bindingdb_kd",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "bindingdb_kd*.csv",
                "columns_required": ["Drug", "Target", "Y"],
            },
            field_mapping={
                "drug_id": "Drug",
                "target_id": "Target",
                "readout": "Y",
                "assay_type": "binding_affinity_kd",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )

        self.registry.register(
            name="tdc_dti_bindingdb_patent",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "bindingdb_patent*.csv",
                "columns_required": ["Drug", "Target", "Y"],
            },
            field_mapping={
                "drug_id": "Drug",
                "target_id": "Target",
                "readout": "Y",
                "assay_type": "binding_affinity_patent",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )

        # DAVIS kinase assay
        self.registry.register(
            name="tdc_dti_davis",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "davis*.csv",
                "columns_required": ["Drug", "Target", "Y"],
            },
            field_mapping={
                "drug_id": "Drug",
                "target_id": "Target",
                "readout": "Y",
                "assay_type": "kinase_davis",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )

        # KIBA score
        self.registry.register(
            name="tdc_dti_kiba",
            template_id="dti_assay_v1",
            match={
                "path_pattern": "kiba*.csv",
                "columns_required": ["Drug", "Target", "Y"],
            },
            field_mapping={
                "drug_id": "Drug",
                "target_id": "Target",
                "readout": "Y",
                "assay_type": "kiba_score",
                "biological_level": "molecular",
            },
            biological_level="molecular",
        )
