# Dargus — Clinical Efficacy Prediction System

Dargus is a clinical-efficacy prediction system for drug-development researchers. It analyzes existing evidence across six biological levels (molecular, cellular, ex vivo, animal, clinical, epidemiological) and translates multi-level evidence into predictions of clinical endpoint effect sizes with 95% confidence intervals.

## Quick start

```bash
pip install -e ".[dev]"
dargus scan-v4 --drugs LRRK2-IN-1 --disease "Alzheimer's disease" --datadir ./my_data
```

Or from a Coding Agent via the Dargus skill:

```
/dargus scan-v4 --drugs LRRK2-IN-1 --disease "Alzheimer's disease" --datadir ./my_data
```

## Architecture

- **D-Base**: sparse-matrix experiment store (`dargus/dbase/`).
- **DBaseManager**: maps raw inputs to D-Base records; the only D-Base writer.
- **DiseaseExpert** & **LevelExperts**: ingest data, curate records, and analyze evidence across six biological levels.
- **Iris**: orchestrates projects, workflows, and agent ensemble.
- **Iris-\***: pluggable prediction agents (`IrisSearch`, `IrisLlm`, `IrisAnalog`, `IrisBayes`, `IrisGnn`, `IrisExpert`).
- **IrisEnsemble**: combines Iris agent predictions with weighted aggregation.

## Design

- `spec.md` — v0.5.0 architecture.
- `.superpowers/plans/` — implementation plans.
- `.superpowers/versions/` — version history and progress.

## Development

```bash
pip install -e ".[dev,llm]"
pytest -q
ruff check dargus tests
black --check dargus tests
```

## Disclaimer

Dargus outputs are for research purposes only and do not constitute clinical advice.
