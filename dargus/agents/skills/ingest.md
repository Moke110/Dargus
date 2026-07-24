---
name: ingest
description: "Ingest new evidence data into D-Base with deduplication and quality checks"
version: "1.0.0"
supported_levels: [molecular, cellular, exvivo, animal, rct, epi]
required_tools: [dbase_write, dbase_query, delegate_to_expert]
input_schema:
  source_path: {type: string, required: true}
  source_type: {type: string, required: false}
output_schema:
  n_records: {type: integer}
  n_duplicates: {type: integer}
  n_errors: {type: integer}
timeout_ms: 600000
fallback: skip
---
# Ingest Skill

## Goal
Ingest new evidence data into D-Base with deduplication and quality checks.

## Workflow
1. Receive task_spec with data source path/URL
2. Explore and parse input files
3. Distribute content to DomainExperts for evidence extraction
4. Each DomainExpert extracts evidence instances via P-R-A loop
5. Call dbase_write for each extracted record
6. Collect DuplicateReviewRequest items
7. Present duplicates for user confirmation
8. Report ingestion summary
