# Dargus v4.0 Agent Roster and Contracts

## Orchestration

### DirectorAgent

- **Responsibilities**: project bootstrap, workflow selection, task dispatch, progress tracking, output aggregation.
- **Does not**: perform analysis, interpret results, modify agent outputs.
- **Entry points**: `start_project`, `status`, `run_workflow`, `run_workflow_v4`, `assign_task`.

## Ingestion

### ReaderAgent

- **Responsibilities**: scan directories, classify files, parse data files, extract experiment instances.
- **Entry points**: `scan_directory`, `parse_data_file`.

### ReportSearcher

- **Responsibilities**: search literature databases, download candidate papers/data, suggest manual downloads.
- **Entry points**: `search`.

### TempRetriever

- **Responsibilities**: map raw inputs or experiment instances to `TemplateRecord`; sole writer to D-Base.
- **Entry points**: `fill_template`, `write_record`.

## Prediction

| Agent | Role | Entry point |
|-------|------|-------------|
| IrisSearch | direct evidence aggregation | `predict` |
| IrisLlm | LLM-based synthesis | `predict` |
| IrisAnalog | analogical reasoning | `predict` |
| IrisBayes | hierarchical Bayesian inference | `predict` |
| IrisGnn | graph neural network prediction | `predict` |
| IrisSelector | choose Iris-* based on data richness | `predict` |
| IrisEnsemble | combine Iris-* outputs | `predict` |

## D-Base Interaction Rules

- Only `TempRetriever` writes D-Base.
- All Iris-* read D-Base only.
- `DirectorAgent` schedules; it does not directly touch D-Base.
