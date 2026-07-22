from dargus.experts.types import ExtractedInstance, ExtractionReport, IngestionSummary


def test_extracted_instance_defaults():
    inst = ExtractedInstance(
        template_id="dti_assay_v1",
        raw_fields={"drug_id": "D1", "target_id": "T1"},
        source_file="data/dti.csv",
        source_row=0,
    )
    assert inst.extraction_confidence == "medium"


def test_extraction_report_counts_instances():
    inst = ExtractedInstance(
        template_id="dti_assay_v1",
        raw_fields={"drug_id": "D1"},
        source_file="data/dti.csv",
        source_row=0,
    )
    report = ExtractionReport(
        level="molecular",
        files_considered=["data/dti.csv"],
        files_selected=["data/dti.csv"],
        source_types={"DTI": 1},
        instances=[inst],
    )
    assert report.n_instances == 1


def test_ingestion_summary_totals():
    report = ExtractionReport(
        level="molecular",
        files_considered=["a.csv"],
        files_selected=["a.csv"],
        source_types={"DTI": 2},
        instances=[
            ExtractedInstance(
                template_id="dti_assay_v1",
                raw_fields={},
                source_file="a.csv",
                source_row=0,
            ),
            ExtractedInstance(
                template_id="dti_assay_v1",
                raw_fields={},
                source_file="a.csv",
                source_row=1,
            ),
        ],
    )
    summary = IngestionSummary(per_level={"molecular": report})
    assert summary.total_instances == 2
    assert summary.template_counts == {"dti_assay_v1": 2}
