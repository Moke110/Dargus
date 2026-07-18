# Dargus — Drug Research Assistant Team

## Description

A 10-agent AI research team plus a full-stack reasoning engine for systematic drug-efficacy hypothesis generation. Dargus analyzes existing data across six biological levels and translates multi-level evidence into clinical endpoint predictions with confidence intervals. It does **not** run wet experiments.

## Quick start

```
/dargus scan LRRK2 "Parkinson's disease"     → target-disease efficacy prediction
/dargus repurpose my_drugs.csv "HCC"         → drug repurposing screen
/dargus translate IC50=50nM "HCC"            → predict clinical outcome from cell data
/dargus review "LRRK2 inhibitors"            → systematic literature review
```

## Agent team

| Agent | Role |
|-------|------|
| DirectorAgent | Project management, task orchestration, progress tracking |
| RetrieverAgent | Unified literature retrieval; spawns SubRetrieverAgents |
| DataMaster | Multi-source data → unified sample-level database |
| MoleculeAgent | Molecular-level analysis (SAR, ADMET, descriptors) |
| CellAgent | Cellular-level analysis (transcriptomics, CRISPR, sensitivity) |
| ExvivoAgent | Ex vivo model analysis (organoids, organ-chips, 3D culture) |
| AnimalAgent | Animal model analysis (translatability, efficacy, toxicity) |
| ClinicAgent | Clinical data analysis (trials, meta-analysis, safety) |
| EpiAgent | Epidemiological analysis (GWAS, MR, rare variants) |
| TranslateAgent | Cross-level translation assessment for a disease |
| Diris | Full-stack reasoning engine: evidence → effect size + CI |

## Installation

```bash
pip install dargus
```

For development:

```bash
pip install -e ".[dev]"
```

## Commands

- `/dargus scan <target> <disease>` — target-disease efficacy prediction
- `/dargus repurpose <drugs> <disease>` — repurposing screen
- `/dargus translate <data> <disease>` — cross-level translation
- `/dargus review <query>` — systematic literature review
- `/dargus agent <name> <task>` — call a specific agent
- `/dargus status <project>` — check project progress
- `/dargus export <project>` — export project outputs

## Configuration

See `dargus/config/dargus_config.yaml`.

## Disclaimer

Outputs are for research purposes only and do not constitute clinical advice.
