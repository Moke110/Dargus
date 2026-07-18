# Dargus — Drug Research Assistant Team

Dargus is a clinical-efficacy prediction system for drug-development researchers. It analyzes existing data across six biological levels (molecular, cellular, ex vivo models, animal, clinical, epidemiological) and translates multi-level evidence into predictions of clinical endpoint effect sizes with confidence intervals.

## Quick start

```bash
pip install -e .
python -m dargus.workflows.target_efficacy_scan --target LRRK2 --disease "Parkinson's disease"
```

Or from a Coding Agent:

```
/dargus scan LRRK2 "Parkinson's disease"
```

## Design

See [`design.md`](./design.md) for the full product design.

## Phase plans

- [`docs/phase0.md`](./docs/phase0.md) — MVP (accepted)
- [`docs/phase1.md`](./docs/phase1.md) — full agent system
- [`docs/phase2.md`](./docs/phase2.md) — calibration and ecosystem
- [`docs/benchdata.md`](./docs/benchdata.md) — recommended benchmark datasets by download order

## Repository layout

```
dargus/
  agents/          # Agent implementations
  database/        # DataMaster schema and converters
  embedding/       # Drug/disease embedding providers
  knowledge/       # Methodology registries and knowledge base
  reasoning/       # Diris full-stack reasoning engine
  tools/           # Shared tools (stats, viz, literature)
  workflows/       # Pre-defined workflows
docs/              # Plans, decisions, norms, progress
tests/             # Test suites
projects/          # Generated project directories (gitignored)
```

## Benchmark datasets

See [`docs/benchdata.md`](./docs/benchdata.md) for a curated list of drug-development benchmark datasets.

To automatically download Tier 1 debugging datasets:

```bash
python scripts/download_tier1.py --output-dir data/benchmarks/tier1
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
black --check .
pytest
```

See [`docs/dev_norms.md`](./docs/dev_norms.md) and [`docs/agent_work_norms.md`](./docs/agent_work_norms.md).

## Disclaimer

Dargus outputs are for research purposes only and do not constitute clinical advice.
