-- Dargus project-level SQLite schema
-- See design.md §5.2 for full specification.

CREATE TABLE IF NOT EXISTS samples (
    sample_id       TEXT PRIMARY KEY,
    sample_type     TEXT NOT NULL,
    species         TEXT,
    strain          TEXT,
    tissue_organ    TEXT,
    cell_type       TEXT,
    cell_line_id    TEXT,
    sex             TEXT,
    age             TEXT,
    disease         TEXT NOT NULL,
    disease_subtype TEXT,
    model_system    TEXT,
    model_induction TEXT,
    genetic_background TEXT,
    treatment       TEXT,
    treatment_dose  REAL,
    treatment_dose_unit TEXT,
    treatment_route TEXT,
    treatment_duration TEXT,
    treatment_regimen TEXT,
    assay_type      TEXT,
    assay_platform  TEXT,
    assay_endpoint  TEXT,
    endpoint_category TEXT,
    source_type     TEXT NOT NULL,
    source_id       TEXT,
    source_url      TEXT,
    source_table_figure TEXT,
    extraction_method TEXT,
    extraction_date TEXT,
    data_quality    TEXT,
    quality_notes   TEXT,
    n_replicates    INTEGER,
    biological_level TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    curator_agent   TEXT,
    curation_date   TEXT,
    data_type       TEXT NOT NULL,
    data_label      TEXT NOT NULL,
    data_value      TEXT NOT NULL,
    data_unit       TEXT,
    data_uncertainty REAL,
    uncertainty_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_samples_disease ON samples(disease);
CREATE INDEX IF NOT EXISTS idx_samples_treatment ON samples(treatment);
CREATE INDEX IF NOT EXISTS idx_samples_level ON samples(biological_level);
CREATE INDEX IF NOT EXISTS idx_samples_assay ON samples(assay_endpoint);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source_type);
CREATE INDEX IF NOT EXISTS idx_samples_project ON samples(project_id);
