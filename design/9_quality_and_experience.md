# Quality & Experience Design

> Dargus must be trustworthy under real-world friction: bad files, missing LLMs, unavailable embeddings, sparse evidence, and long-running benchmarks. The quality layer defines how the system behaves when things go wrong and how it proves it is working.

## Testing norms

- **Test one module per file.** `test_<component>_<behavior>` naming.
- **TDD.** Write the failing test first, confirm it fails, then implement.
- **Real filesystems, fake externals.** Use `tmp_path` for file tests. Mock only external services (LLM APIs, PubMed).
- **Real D-Base state.** D-Base tests create actual records and files rather than mocking the store.

## Required test coverage for v1.0.0

- D-Base exact and semantic deduplication.
- Embedding tool dimensionality, symmetry, and missing-embedding fallback.
- Semantic read with metadata filters and top-k sorting.
- Ingest end-to-end: directory → records → D-Base.
- Predict end-to-end: query → Expert reports → FinalReport.
- Benchmark holdout marking and restoration.
- CLI command parsing and handler wiring.
- Expert `extract()` and `assess()` from fixtures.
- Iris multi-round convergence.
- Hook chain integration.

## Error handling and degradation

| Scenario | Behavior |
|---|---|
| LLM unavailable | Return stub response; log warning; continue if possible |
| Embedding tool unavailable | Records are still written; semantic search skips records without embeddings; exact dedup still works |
| Single Expert fails | Other Experts continue; failure noted in report |
| Empty D-Base | Predict returns a report with `confidence_level: insufficient_data`, DES/DCS unset, and a warning |
| Duplicate review request | Pause for confirmation; default to allow if no callback |
| Validation failure | Skip record; log reason; continue with remaining records |
| Benchmark matches zero records | Abort with clear message |
| `max_rounds` reached | `SafetyNetHook` forces convergence and flags `max_rounds_reached` |

## Quality gates

| Gate | Target |
|---|---|
| Unit tests | `pytest -q` passes |
| Linting | `ruff check dargus tests` clean |
| Formatting | `black --check dargus tests` passes |
| Traceability | 100% of delivered predictions cite supporting records (`insufficient_data` reports excepted) |
| Write safety | 100% of evidence writes go through the single-writer D-Base API |
| Test-set leakage | 0 holdout records read during Benchmark inference |
| Report validation | Format, DES/DCS range, and supporting-record checks pass for every delivered report |

## Observability

Every Agent run produces an `AgentReport` with a `CallTrace`. Every workflow produces a result dict with status and rounds completed. Tool calls can be audited via `ToolAuditHook`. These artifacts make it possible to answer “Why did Dargus predict this?” after the fact.

## v1.0.0 scope

- Test-driven development with module-level test files.
- End-to-end tests for Ingest, Predict, and Benchmark.
- Safety-net hooks for round limits and timeouts.
- Report validation enforcing the prediction contract.
