# Dargus Agent Roster and Contracts

## Management agents

### DirectorAgent

- **Responsibilities**: project bootstrap, workflow selection, task dispatch, progress tracking, output aggregation.
- **Does not**: perform analysis, interpret results, modify agent outputs.
- **Entry points**: `start_project`, `status`, `run_workflow`, `assign_task`.

### RetrieverAgent

- **Responsibilities**: unified literature search, PDF extraction, structured data extraction.
- **Entry points**: `search`, `extract`, `load_library`.

## Data agents

### DataMaster

- **Responsibilities**: create project database, ingest heterogeneous sources, normalize metadata, query data.
- **Entry points**: `ingest`, `query`, `get_summary_stats`.

## Analysis agents

Each analysis agent handles one biological level and produces the standard five-pack.

| Agent | Level | Entry point |
|-------|-------|-------------|
| MoleculeAgent | molecular | `dargus_molecular_analyze` |
| CellAgent | cellular | `dargus_cellular_analyze` |
| ExvivoAgent | exvivo | `dargus_exvivo_analyze` |
| AnimalAgent | animal | `dargus_animal_analyze` |
| ClinicAgent | clinical | `dargus_clinical_analyze` |
| EpiAgent | epidemiology | `dargus_epidemiology_analyze` |

## Translation agent

### TranslateAgent

- **Responsibilities**: assess disease-specific cross-level translation reliability.
- **Entry point**: `dargus_translate_assess`.

## Reasoning engine

### Diris

- **Responsibilities**: predict normalized clinical effect sizes and CIs from embeddings and level evidence.
- **Entry point**: `dargus_reasoning_predict`.

## Standard five-pack

Every analysis agent writes:

1. `report.md`
2. `figures/*.png`
3. `data/*.csv`
4. `code/analysis.py`
5. `level_embedding.json`

See `docs/agent_work_norms.md` for details.
