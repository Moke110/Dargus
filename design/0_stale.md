# Stale — Changed & Abandoned Designs

> This document records names, norms, and designs that were changed or abandoned while the design docs evolved. It exists so that a reader who encounters an old term — in stale-present sheets, historical notes, or an old checkout — can find what replaced it. Each entry points to the doc that carries the current design.
>
> Documentation norm: design docs describe only what ships in v1.0.0. Deferred features live in the `Out of Scope` section at the end of the corresponding doc — never in the body. Historical comparisons live here, not in the design docs.

## Documentation norms

### 2026-07-27 — Prospect and design-changes docs folded into per-doc sections

- **`x_prospect.md` → distributed.** The standalone prospect doc was deleted. Each deferred item now lives in the `Out of Scope` section at the end of the design doc that owns the topic.
- **`0_design_changes.md` → this file.** The dated design-changes log was deleted. Historical "what changed and why" is preserved here as stale entries; docs no longer carry inline historical comparisons.
- **Benchmark and Knowledge system moved out of v1.0.0 scope.** The v1.0.0 goal narrowed to finishing the Ingest and Predict workflows end-to-end with no accuracy demand. Both were stripped from doc bodies and recorded as Out of Scope (`7_workflows.md`, `6_skills_tools_knowledge.md`).

## Workflow designs

### Train / Infer / Benchmark → Ingest / Predict

The earliest workflow split was **Train / Infer / Benchmark** with a global D-Base. In the v1.0.0 design this became: **Ingest** (new — raw data to evidence), **Predict** (renamed from Infer), and **Benchmark** (kept, then deferred out of v1.0.0 scope). **Train** (fine-tuning, calibration, learning delegation policies) is deferred, not abandoned — see the Out of Scope section of `7_workflows.md`. Current design: `7_workflows.md`.

## Abandoned names

| Abandoned | Current | Where |
|---|---|---|
| Planner-Executor-Critic harness | Perceive → Reason → Act (PRA) loop | `4_agents.md` |
| 95% confidence interval output | DES ± DCS (`efficacy_score`, `confidence_score`) | `4.1_agent_protocols.md` |
| `RuntimeContext` | `DargusRuntime` | `3_runtime.md` |
| `DBaseManager` | single-writer D-Base API, reached through tools | `2_D-Base.md` |
| `EmbeddingService` | `embedding` Tool with session `ToolCache` | `6_skills_tools_knowledge.md` |
| `AcceptanceGateHook` | `ReportValidationHook` | `5_hooks.md` |
| `FourDExpert` | `D4Expert` | `4_agents.md` |
| "TUI" | "CLI" (one-shot commands + REPL) | `8_cli.md` |
| `y.basis`, `y.ci95`, `phenotypes`, `experiment_group_id`, `simulation_provenance`, Exposure field group | `y.to_basis`, `y.dispersion` with `CI95`, `bg.phenotype`, `related_evidence_id`, `source_entry`/`source_time`, dose/duration in Background group | `2.1.1_D-Base_field.md`, `2.1.2_D-Base_enumerate.md` |
| `spec.md`, `roadmap.md` | the `design/` doc set | this directory |

## Abandoned designs

### MCP adapter (v0.19.0)

The `dargus/adapters/mcp/` adapter was removed from the repository. An MCP server exposing `dargus.api` is a deferred feature — see the Out of Scope section of `8_cli.md`.

### cadev dev-workflow (2026-07-26)

The cadev orchestration layer for development (`.claude/skills/cadev/`, `docs/cadev/`, the `cadev-state-check.sh` hook) was removed entirely. Development no longer routes through `/cadev`; quality checks are enforced by the pre-commit gate documented in `CLAUDE.md`.

### "Graceful degradation" design principle (2026-07-25)

Removed from the design principles: it is an implementation quality, not a principle. The failure behaviors it covered remain in the error-handling table of `9_quality_and_experience.md`.

### In-record mutable fields (2026-07-26)

Early designs kept `status`, `superseded_by`, the LLM summary, and the embedding inside the evidence record, contradicting the append-only invariant. These are now three append-only **sidecar fields** keyed by `evidence_id`, outside the immutable 50-field record — see `2.2_D-Base_storage_and_lifecycle.md`.
