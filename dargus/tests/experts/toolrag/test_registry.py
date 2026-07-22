import tempfile
from pathlib import Path

from dargus.experts.toolrag.registry import ConverterRegistry


def test_registry_matches_by_path_pattern_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "bindingdb_ic50.csv").write_text("ID1,ID2,Y\nD1,T1,5.0\n")

        registry = ConverterRegistry()
        registry.register(
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
            },
            biological_level="molecular",
        )

        instances = registry.convert_file(data_dir / "bindingdb_ic50.csv")
        assert len(instances) == 1
        assert instances[0].raw_fields["drug_id"] == "D1"
        assert instances[0].raw_fields["readout"] == 5.0
        assert instances[0].raw_fields["assay_type"] == "binding_affinity"
