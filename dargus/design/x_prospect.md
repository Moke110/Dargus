# Prospect — Beyond v1.0.0

> This document collects longer-term directions for Dargus after v1.0.0. Nothing here is required for the v1.0.0 release; it exists to keep the core design clean while preserving a place for future vision.

## Multi-model prediction ensemble

v1.0.0 Predict is Expert-driven. A future release can add specialized prediction primitives:

- `iris_search` — literature and database search agent,
- `iris_analog` — analog-based reasoning,
- `iris_bayes` — hierarchical Bayesian modeling,
- `iris_gnn` — graph-neural-network prediction,
- `iris_llm` — direct LLM reasoning.

An `Iris.ensemble()` layer could aggregate these models with weights derived from calibration and evidence coverage.

## Advanced Routing Skills

The v1.0.0 Routing Skill is field match + semantic search. Future Routing Skills could include:

- **Expert knowledge graph** — each Expert maintains a graph of drug-target-disease relationships and traverses it to find indirect evidence,
- **Contrastive retrieval** — retrieve evidence that challenges the current hypothesis,
- **Temporal retrieval** — prefer recent evidence or track evidence evolution.

Additional biomedical Tools and domain-specific Skills are also deferred.

## Per-Agent model routing

v1.0.0 uses a single fixed reasoning model for all Agents. A future release could route different Agents (or different PRA phases) to different models via a runtime model router.

## Training pipeline

A future Train workflow would support:

- fine-tuning local LLMs on Dargus-produced Expert assessments,
- calibrating Bayesian and GNN models on held-out records,
- learning delegation policies from historical convergence traces.

Train is deferred until local-model deployment and post-v1.0.0 model integration are stable.

## Knowledge graph and heterogeneous evidence

D-Base v1.0.0 is a flat keyed-object store. A future release could layer a heterogeneous knowledge graph on top, linking drugs, targets, pathways, diseases, and trials while keeping D-Base as the authoritative evidence source.

## External interfaces: API and MCP server

Dargus v1.0.0 ships CLI and REPL entry points only. A programmatic API and an optional MCP server could expose Dargus Tools and workflows to external callers and agent platforms, letting Claude Code or other agents query D-Base and run Predict through a standardized protocol.

## Benchmark validation split

The `holdout-valid` status reserves records for a validation set (e.g., hyperparameter or prompt tuning) separate from the test set. v1.0.0 Benchmark uses only `holdout-test`.

## Schema extension

`D-Base.add_field()` — a researcher-facing function to extend the record schema with new fields — requires a formal migration path across validation, embedding, and vocabulary registries.

## Operational maturity

Post-v1.0.0 operations could include:

- environment separation (dev / staging / prod),
- structured audit logs and metrics,
- streaming Predict output,
- prompt caching and session reuse,
- scheduled or event-driven ingestion,
- always-allow / always-ask permission tiers.

## Note on scope

These items are intentionally excluded from v1.0.0. The goal of the first major release is to prove that the Expert-driven, D-Base-backed, DES ± DCS prediction system works end-to-end. Everything else builds on that foundation.
