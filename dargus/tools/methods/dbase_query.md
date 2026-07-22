---
tool: dbase_query
level: any
assay_type: evidence_retrieval
database_source: "D-Base (internal evidence store)"
key_columns: [drug_id, disease_id, biological_level, readout_value, evidence_id]
limitations: "Only returns records already ingested into D-Base; coverage depends on ingest runs"
---
# D-Base Query

## Method
Structured query against the internal evidence store (D-Base).
Returns evidence records matching drug, disease, and biological level filters.

## Output Interpretation
- `evidence_id`: content-addressed unique identifier (sha256 prefix)
- `readout_value`: primary effect measurement
- `biological_level`: evidence tier (molecular -> rct)
