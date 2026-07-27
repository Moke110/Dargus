# Design Changes Log

This file records design changes and the rationale behind them. It keeps the main design documents free of historical comparisons.

## 2026-07-26 — cadev workflow removed

**Changed docs:** `CLAUDE.md`, `.gitignore`, `.claude/hooks/pre-commit-gate.sh`, `0_design_changes.md`

**What changed:**
- The cadev dev-workflow skill was removed entirely: `.claude/skills/cadev/`, its state/config/knowledge directory `docs/cadev/` (including `cadev-principles.md` and `references/`), and the `.claude/hooks/cadev-state-check.sh` hook.
- `.claude/hooks/pre-commit-gate.sh` no longer reads cadev state to enforce quality checks; it now only enforces the no-commits-on-main rule.
- `CLAUDE.md` no longer routes development through `/cadev` and no longer points at `docs/cadev/` knowledge files.
- `.gitignore` no longer excludes `docs/cadev/state.json`.

**Why:**
The cadev orchestration layer is no longer used to manage development. Historical mentions of cadev in `docs/history-specs/` and `dargus/version.md` are preserved as-is since they are historical records. The knowledge summaries previously held in `docs/cadev/references/` are superseded by the design docs in `dargus/design/`.

## 2026-07-26 — Sidecar fields, entry-point scope, CLI consolidation, and insufficient-data semantics

**Changed docs:** `1_design_overview.md`, `2_D-Base.md`, `2.1_D-Base_field_vocabulary.md`, `2.1.1_D-Base_field.md`, `2.1.2_D-Base_enumerate.md`, `2.2_D-Base_storage_and_lifecycle.md`, `3_runtime.md`, `4.1_agent_protocols.md`, `5_hooks.md`, `6_skills_tools_knowledge.md`, `7_workflows.md`, `8_cli.md`, `9_quality_and_experience.md`, `x_prospect.md`, `CONTEXT.md`

**What changed:**
- D-Base now has three sidecar fields stored outside the evidence record, each in its own append-only file keyed by `evidence_id`: lifecycle `status` with `superseded_by` (`sidecars/status.jsonl`), the LLM summary (`sidecars/llm_summary.jsonl`), and the embedding (`sidecars/embeddings-{model_fp}.jsonl`). `status` and `superseded_by` moved out of the record; the record schema is now 50 fields. Sidecar fields never participate in the identity hash.
- Status changes are lifecycle transitions appended to the status sidecar (latest entry wins; no entry means `active`), resolving the contradiction between "no record is mutated in place" and Benchmark holdout flips / retraction.
- v1.0.0 entry points are CLI and REPL only. The programmatic API and MCP server moved to `x_prospect.md`; the runtime's components table, Mermaid diagram, and health-flag behavior now mention only CLI/REPL.
- Runtime health: the runtime goes unhealthy only on unrecoverable dependency failure; the undefined "inconsistent shared state" condition was removed.
- CLI consolidation: removed `dargus model`, `dargus clear-dbase`, the `dargus embedding *` commands, and REPL `/model` and `/clear-dbase`. `dargus config` (CLI) and `/config` (REPL) are the single entry point for configuring the reasoning model and other settings; detailed capabilities are left to implementation.
- Added `dbase_update_status` and `dbase_write_summary` tools so sidecar writes are first-class capabilities of the single-writer layer.
- Empty-D-Base Predict output is now a report with `confidence_level: insufficient_data` and DES/DCS unset; the supporting-record requirement has an explicit `insufficient_data` carve-out in `4.1_agent_protocols.md` and `5_hooks.md`.
- Ingest's Explore phase names Iris as the coordinator Agent.
- `CONTEXT.md` glossary updated: Perceive → Reason → Act loop, DES ± DCS output, and the y-axis description no longer mentions "confidence interval".
- Fixed stale leftovers: "sparse matrices" in testing norms, `drug_id`/`y_type` filter names in the Routing Skill (now `bg.drugs` entity IDs, `bg.disease_id`, `y.type`).
- Post-v1.0.0 visions consolidated into `x_prospect.md` (per-Agent model routing, API/MCP, Benchmark validation split, schema extension, advanced routing/tools/skills); other docs no longer carry "(post-v1.0.0)" markers.
- The project knowledge summaries formerly kept under `docs/cadev/` were resynced to the design docs (they no longer referenced `spec.md`/`roadmap.md` (both deleted), `DBaseManager`, `EmbeddingService`, `AcceptanceGateHook`, 95% CI output, `Planner/Executor/Critic` phases, or `FourDExpert`); those summaries have since been removed together with cadev (see the 2026-07-26 cadev removal entry above).

**Why:**
A doc-wide consistency review (grill-with-docs session) found cross-doc contradictions (MCP/API listed as runtime entry points while deferred in the overview; record immutability vs. status flips), stale references from earlier designs, and undefined terms. The sidecar model makes the append-only invariant uniformly true: evidence records are immutable, and all mutable lifecycle/derived state lives in append-only sidecars keyed by `evidence_id`. Consolidating configuration under `dargus config`/`/config` keeps the CLI surface small while leaving room to grow.

## 2026-07-26 — Doc renumbering: Agents ↔ Hooks swap

**Changed docs:** `5_agents.md` → `4_agents.md`, `5.1_agent_protocols.md` → `4.1_agent_protocols.md`, `4_hooks.md` → `5_hooks.md`

**What changed:**
- Renumbered the three doc files so Agents (and Agent Protocols) sit at section 4 and Hooks at section 5; updated the filename reference in `3_runtime.md` accordingly.

**Why:**
Groups the agent design docs together and places Hooks after them to match the intended reading order.

## 2026-07-26 — Runtime doc: components table and Mermaid relation map

**Changed docs:** `3_runtime.md`, `0_design_changes.md`

**What changed:**
- Added a `Components` subsection under `DargusRuntime` in `3_runtime.md`, listing every runtime-controlled component and what the runtime does with it.
- Added a Mermaid `flowchart TB` diagram showing relations between `DargusRuntime`, user-facing entry points, runtime-owned singletons, `AgentFactory`, Agents, Tools/Skills/Knowledge, D-Base, `ToolCache`, and `HookRegistry`.

**Why:**
A plain bullet list makes it easy to miss how the components interact. The table plus diagram makes ownership, dependency, and data flow explicit.

## 2026-07-26 — Embedding tool and CLI commands

**Changed docs:** `6_skills_tools_knowledge.md`, `8_cli.md`

**What changed:**
- Added the `embedding` Tool to the core tool list in `6_skills_tools_knowledge.md`; it supports `embed`, `test`, and `info` operations and loads the project-level HuggingFace model into the session `ToolCache`.
- Updated the Routing Skill description to use the active embedding-model fingerprint table rather than a stored embedding inside each record.
- Added CLI commands `dargus embedding status`, `dargus embedding download`, `dargus embedding test`, and `dargus embedding set-model` to `8_cli.md`.
- Updated the configuration flow and v1.0.0 scope in `8_cli.md` to cover embedding-model selection, download, test, and re-embedding.

**Why:**
Moving the embedding model management into a first-class Tool and CLI surface makes model selection explicit, keeps the embedding model resident during ingest via `ToolCache`, and lets users change models and re-embed without touching the evidence schema.

## 2026-07-26 — D-Base: embeddings moved out of the main evidence record

**Changed docs:** `2_D-Base.md`, `2.1_D-Base_field_vocabulary.md`, `2.1.1_D-Base_field.md`, `2.1.2_D-Base_enumerate.md`, `2.2_D-Base_storage_and_lifecycle.md`

**What changed:**
- Removed the `embedding` field from the D-Base evidence record.
- Updated the design-doc field count from 53 to 52.
- Evidence embeddings are now stored in a separate append-only table (`embeddings/embeddings-{model_fp}.jsonl`) keyed by `evidence_id` and the active embedding-model fingerprint, with an `embeddings/manifest.json` tracking active fingerprints.
- Added a re-embedding process that generates vectors for a new model without touching the authoritative evidence shards.
- Updated the write lifecycle to generate embeddings via the embedding tool and append them to the embeddings table.
- Updated recovery semantics: if embedding generation fails, the record is still written; semantic search skips records that lack a vector for the active model.

**Why:**
Decoupling embeddings from the evidence record lets users change the project-level embedding model and re-embed all evidence without altering evidence identity or invalidating the append-only store. It also keeps the evidence schema stable while the embedding model evolves.

## 2026-07-26 — Runtime & Hooks design refinements

**Changed docs:** `3_runtime.md`, `5_hooks.md`, `4.1_agent_protocols.md`, `7_workflows.md`, `9_quality_and_experience.md`, `0_design_changes.md`

**What changed:**
- `DargusRuntime` now explicitly holds `AgentFactory`, CLI/REPL/API/MCP entry-point packaging, and a session-scoped `ToolCache`.
- The v1.0.0 reasoning model is a single fixed LLM; the model router was removed from the runtime description and noted as future scope.
- Added health-flag semantics: healthy at startup, unhealthy on unrecoverable dependency failure or inconsistent shared state; entry points refuse new sessions until restart.
- Added symmetric hook points `PERCEIVE_END`, `REASON_START`, `ACT_START`.
- Added report hook points `DOMAIN_REPORT_PRODUCED` and `D4_REPORT_PRODUCED`.
- Renamed `AcceptanceGateHook` to `ReportValidationHook`; it now validates format, DES/DCS ranges, and evidence_id existence at every intermediate report and at `SESSION_END`, and sets a `report_valid` flag checked by deliver-report tools.
- `SafetyNetHook` now enforces `max_rounds`, `round_timeout`, and `session_timeout`; the minimum-evidence-coverage rule was removed.
- Defined a mutable hook context with reserved keys and documented exception-based veto semantics (fail-closed default; observer-only hooks fail-open).
- Specified hook registration order, default registration by the runtime, and structured-log observability.
- Rewrote the “Why hooks instead of hardcoded workflows” paragraph to remove the historical comparison and moved that historical note into this log.
- Cross-checked `4.1_agent_protocols.md`, `7_workflows.md`, and `9_quality_and_experience.md` and replaced remaining `AcceptanceGateHook`, `EmbeddingService`, and coverage references to match the new runtime/hooks model.

**Why:**
The runtime doc had internal gaps (missing AgentFactory/entry points, ambiguous model routing) and the hook model needed to support iterative report validation in the Predict workflow. Moving historical rationale into this log and updating related docs keeps the design set internally consistent.

## 2026-07-26 — Split Runtime & Hooks into separate design docs

**Changed docs:** `3_runtime.md` (new), `5_hooks.md` (new), `3_runtime_and_hooks.md` (deleted), `0_design_changes.md`

**What changed:**
- Split `3_runtime_and_hooks.md` into two focused documents.
- `3_runtime.md` covers `DargusRuntime`, reasoning model, `ToolCache`, health flag, and runtime v1.0.0 scope.
- `5_hooks.md` covers hook philosophy, hook context, hook points, core hooks (`SessionInitHook`, `SkeletonContextHook`, `ToolAuditHook`, `SafetyNetHook`, `ReportValidationHook`, `ResultReportHook`), hook registration/execution semantics, and hooks v1.0.0 scope.
- Added cross-references between the two new docs.
- Updated historical "Changed docs" entries in this log to point to the new filenames.

**Why:**
Runtime lifecycle/configuration and hook semantics are separate concerns. Splitting them lets each doc stay focused and makes it easier to evolve hooks (new points, new core hooks) without revising runtime internals, and vice versa.

## 2026-07-25 — Prediction output: from 95% CI to DES ± DCS

**Changed docs:** `1_design_overview.md`, `4.1_agent_protocols.md`, `7_workflows.md`, `9_quality_and_experience.md`

**What changed:**
- Dargus no longer reports predictions as a 95% confidence interval.
- Predictions are now reported as a **Dargus efficacy score (DES)** plus a **Dargus confidence score (DCS)**, together expressed as **DES ± DCS**.
- `FinalReport` fields changed from `efficacy_low` / `efficacy_up` to `efficacy_score` (DES) / `confidence_score` (DCS).

**Why:**
A 95% confidence interval has a precise frequentist statistical definition. Dargus cannot compute that from the heterogeneous evidence it consumes, so it reports its own efficacy and uncertainty metrics instead of borrowing a term with a stricter meaning.

## 2026-07-25 — Agent harness: from Planner-Executor-Critic to Perceive-Reason-Act

**Changed docs:** `1_design_overview.md`, `4_agents.md`, `4.1_agent_protocols.md`

**What changed:**
- The universal Agent harness is now described as **Perceive → Reason → Act**.
- `CallTrace.phase` values changed from `planner | executor | critic` to `perceive | reason | act`.

**Why:**
Perceive-Reason-Act is the standard framing for agentic systems in the broader ecosystem. Adopting it makes the design easier to understand for readers familiar with agent literature and avoids implying a hard-coded three-stage pipeline.

## 2026-07-25 — Top-level architecture: one-line relation tree

**Changed docs:** `1_design_overview.md`

**What changed:**
- The architecture diagram was replaced with a one-line relation tree: `Dargus CLI/REPL/API/MCP - Iris - RuntimeContext (- HookRegistry) - Skills (- Predict, Ingest, Benchmark) - Agents (- Domain Experts, D4Expert) - Tool / Skill / Knowledge layer - D-Base`.
- A short prose section explains the call/delegation/import/management relation for each link.

**Why:**
The previous box diagram showed stacking but not the nature of the relationships between components. The tree format makes the call chain and side branches explicit.

## 2026-07-25 — Removed "Graceful degradation" from design principles

**Changed docs:** `1_design_overview.md`, `9_quality_and_experience.md`

**What changed:**
- The "Graceful degradation" design principle was removed from `1_design_overview.md`.
- The matching v1.0.0 scope bullet was removed from `9_quality_and_experience.md`.

**Why:**
Graceful degradation is an implementation quality, not a design principle. The specific failure behaviors it covered are still documented in the error-handling table in `9_quality_and_experience.md`.

## 2026-07-25 — Top-level runtime: from RuntimeContext to DargusRuntime

**Changed docs:** `1_design_overview.md`, `3_runtime.md`, `5_hooks.md`

**What changed:**
- `RuntimeContext` was renamed to **DargusRuntime**.
- DargusRuntime is now described as the program container registered with the OS task manager, packaging and managing CLI/REPL/API/MCP entry points, HookRegistry, AgentFactory, and the D-Base store.
- Iris is no longer shown as the root of the architecture; it is an Agent created by AgentFactory.
- All Agents (Iris, Domain Experts, D4Expert) are siblings under AgentFactory. Iris dispatches tasks to other Agents through agent communication protocols; it does not manage their lifecycle.
- D-Base is accessed through the Tool/Skill/Knowledge layer, not through a separate D-BaseManager.

**Why:**
Iris is an Agent and uses the Perceive-Reason-Act harness, which is flexible by design. The overall Dargus program needs a stable top-level manager for lifecycle, hooks, and dependency injection. DargusRuntime provides that certainty while Iris remains the top-level orchestrator Agent.

## 2026-07-25 — Architecture diagram: real 2D tree

**Changed docs:** `1_design_overview.md`

**What changed:**
- The one-line relation tree was replaced with an ASCII 2D tree showing parent/child and ownership relations.

**Why:**
A flat one-line tree does not clearly express nesting and ownership. The 2D tree makes the hierarchy explicit.

## 2026-07-25 — D-Base field refinements: identity, source tracking, y-axis, and background

**Changed docs:** `2.1_D-Base_field_vocabulary.md`, `2.1.1_D-Base_field.md`, `2.1.2_D-Base_enumerate.md`, `2_D-Base.md`

**What changed:**
- Renamed `experiment_group_id` to `related_evidence_id`, changed it to a list of strings, and moved it from Identity to Metadata.
- Moved `sources` from Metadata to Identity.
- Changed `sources` structure to `[{rank, type, name}]` with `type` as an enum of source categories.
- Added `source_entry` (accession/DOI/PMID/version) and `source_time` (update/publication time) as Identity fields.
- Changed `xy.count` descriptive shape from `0` to `1`.
- Renamed `y.basis` to `y.to_basis` and updated its enum to `absolute`, `change`, `fold_change`, `log2_fold_change`, `log10_fold_change`.
- Removed `y.ci95`; confidence intervals are represented through `y.dispersion` with `type: CI95`.
- Renamed `phenotypes` to `bg.phenotype` and moved it to the Background group.
- Removed `simulation_provenance`.
- Removed inline historical comparisons from `2.1.1_D-Base_field.md`.
- Added a `y.effect.type` selection guide in `2.1.2_D-Base_enumerate.md` based on `biological_level`, `evidence_design`, and `clinical_design.comparator_type`.
- Changed `y.effect` schema from `{value, type, ci95?, scale?}` to `{value, value_type, dispersion, dispersion_type}` and documented `dispersion_type` values.
- Replaced an outdated "confidence-interval prediction system" reference in `x_prospect.md` with "DES ± DCS prediction system".

**Why:**
Source identity belongs at the top level because provenance is part of what makes a record distinct. `source_entry` and `source_time` make the primary source reference precise and time-aware. `xy.count = 1` for descriptive records aligns the count with the actual number of data points. `y.to_basis` uses shorter names and adds log-fold variants common in omics. Removing `y.ci95` avoids redundancy with `y.dispersion`. `bg.phenotype` places phenotype terms with other background context. `simulation_provenance` was removed because simulation model and version are now captured in `source_entry` and `source_time`.

## 2026-07-25 — D-Base field vocabulary reorganization and reference docs

**Changed docs:** `2.1_D-Base_field_vocabulary.md`, `2_D-Base.md`, `2.1.1_D-Base_field.md` (new), `2.1.2_D-Base_enumerate.md` (new), `2.2_D-Base_storage_and_lifecycle.md`, `6_skills_tools_knowledge.md`, `7_workflows.md`, `9_quality_and_experience.md`

**What changed:**
- Added `age` to the Sample group.
- Moved `exvivo_platform` from the Exposure group to the Sample group.
- Renamed `assay_platform` to `y.assay` and moved it to the y-axis group.
- Renamed `exposure_dose_value`, `exposure_dose_unit`, `exposure_duration_value`, `exposure_duration_unit` to `bg.dose_value`, `bg.dose_unit`, `bg.duration_value`, `bg.duration_unit` and moved them to the Background group.
- Removed the Exposure group.
- Made the Clinical group explicit by listing every `clinical_design.*` field instead of `clinical_design.*`.
- Schema count updated from 52 to 53 fields.
- Created `2.1.1_D-Base_field.md` with field-by-field semantics.
- Created `2.1.2_D-Base_enumerate.md` documenting all controlled vocabularies and explaining important ones.
- Updated the D-Base field reference summary to match the new field names and 53-field count.

**Why:**
The Exposure group mixed sample-source, background, and outcome information. Moving `exvivo_platform` to Sample, dose/duration to Background, and assay platform to the y-axis makes the three-axis model consistent. Adding `age` captures a common sample descriptor. Splitting the concise vocabulary doc from detailed field and enumeration references makes the docs usable for both quick lookup and deep reading.

## 2026-07-25 — D-Base principles, human-owned duplicate review, and post-v1.0.0 add_field

**Changed docs:** `2_D-Base.md`, `6_skills_tools_knowledge.md`, `7_workflows.md`, `9_quality_and_experience.md`

**What changed:**
- Added four D-Base principles to `2_D-Base.md`: one variable per evidence, one intact evidence per input, LLM summary as fallback, and expandable vocabularies under human approval.
- Clarified that the final decision on semantic duplicate review is made by the human user, not by the store, Agents, or an automated workflow.
- Added `D-Base.add_field()` to the out-of-scope list in `2_D-Base.md`.
- Removed remaining `DBaseManager` references in `6_skills_tools_knowledge.md` and `9_quality_and_experience.md`, replacing them with "single-writer D-Base API" wording.
- Replaced the remaining "Critic" reference in `7_workflows.md` with "Reason step".

**Why:**
Principles make the evidence model explicit and constrain how D-Base grows. Human ownership of duplicate review matches the human-in-the-loop design principle and prevents an Agent from silently accepting or rejecting near-duplicates. `add_field()` is deferred because schema extension needs migration design. Cleaning up leftover `DBaseManager` and `Critic` terms keeps the docs consistent with the v0.17.0 architecture and PRA harness changes.

## 2026-07-25 — Design principle: multi-domain expert cooperation

**Changed docs:** `1_design_overview.md`

**What changed:**
- Added a design principle: "Multi-domain cooperation. Domain Experts with complementary specializations delegate, assess, and synthesize evidence across molecular, cellular, animal, and clinical levels. No single Expert has to reason outside its domain."

**Why:**
Cross-level collaboration is central to Dargus's predictive value. Making it a first-class design principle clarifies why the system is organized as multiple specialized Experts rather than one monolithic model.

## v0.19.0 — MCP adapter removed

The MCP adapter (`dargus/adapters/mcp/`) has been removed from the repository.
It is planned as a post-v1.0.0 feature. The `dargus/adapters/` directory has
been removed entirely since MCP was the only adapter present.

See: https://github.com/Moke110/Dargus/issues/3
