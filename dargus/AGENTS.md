# Dargus 0.5.0 Agent Roster and Contracts

## Orchestration

### Iris (Commander)

- **Responsibilities**: project lifecycle management, workflow orchestration, agent ensemble.
- **Does not**: perform analysis, interpret results, modify agent outputs.
- **Entry points**: `start_project`, `status`, `ingest_project`, `plan_prediction`, `predict`, `ensemble`.

## Ingestion & Expertise

### DiseaseExpert

- **Responsibilities**: data ingestion (file parsing, level dispatch), prediction planning (PlanProposal with human confirmation), final prediction (coordinating LevelExperts).
- **Entry points**: `ingest`, `plan`, `predict`.

### {Level}Expert (×6)

- **MolecularExpert / CellularExpert / ExvivoExpert / AnimalExpert / ClinicalExpert / EpiExpert**
- **Responsibilities**: curate experiment instances (validate level, write via DBaseManager), analyze evidence within level.
- **Entry points**: `curate`, `analyze`.

## Data Guardianship

### DBaseManager

- **Responsibilities**: sole D-Base read/write entry point. Single-record writes, immediate persistence.
- **Entry points**: `read_records`, `read_record`, `write_record`, `fill_template`, `query`, `list_templates`, `get_template`.

## Prediction

| Agent | Role | Entry point |
|-------|------|-------------|
| IrisSearch | direct evidence aggregation (Beta posterior) | `predict` |
| IrisLlm | LLM-based synthesis | `predict` |
| IrisAnalog | analogical reasoning (similarity-weighted) | `predict` |
| IrisBayes | hierarchical Bayesian inference (PyMC) | `predict` |
| IrisGnn | graph neural network prediction (R-GCN) | `predict` |
| IrisExpert | expert system bridge (wraps DiseaseExpert) | `predict` |
| IrisEnsemble | weighted aggregation of Iris-* outputs | `predict` |

## D-Base Interaction Rules

- Only `DBaseManager.write_record()` writes D-Base — single complete `TemplateRecord` at a time, immediately persisted.
- All Iris-* read D-Base only.
- Iris orchestrates; it does not directly touch D-Base.
