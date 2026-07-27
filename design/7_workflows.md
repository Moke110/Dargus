# Workflow Design

> Dargus v1.0.0 has two user-facing workflows: **Ingest** and **Predict**. Each is implemented as a Skill-driven, hook-orchestrated process rather than a hardcoded script.

## Ingest — from raw data to structured evidence

Ingest turns files, tables, and documents into validated D-Base records. It is the only way new evidence enters the system.

### Phases

1. **Explore.** Iris, as coordinator, scans the source directory and classifies files by domain (molecule, biomedical, bioinformatics, clinical).
2. **Convert.** Each Domain Expert reads the files in its domain and extracts structured evidence instances.
3. **Input.** Each instance is validated, deduplicated, embedded, and written to D-Base via the single-writer API.

### Human-in-the-loop

After conversion, Dargus presents an `IngestionSummary` and any `DuplicateReviewRequest` soft flags. The user (or `confirm_callback`) decides whether to proceed, skip duplicates, or abort. Only after confirmation are records written.

## Predict — from evidence to efficacy scores

Predict estimates the probability that a drug improves outcomes for a disease across one or more endpoints.

### Phases

1. **Intent validation.** Iris confirms drug IDs, disease ID, and endpoints.
2. **Expert dispatch.** Each Domain Expert uses its Routing Skill to retrieve relevant evidence from D-Base.
3. **Assessment.** Experts produce `ExpertReport`s with evidence quality, limitations, and delegations.
4. **Delegation / re-assessment.** If an Expert delegates records to another Expert, the loop continues until convergence or `max_rounds`.
5. **Synthesis.** D4Expert combines all reports into a `FinalReport`.
6. **Report validation.** `ReportValidationHook` checks every `ExpertReport` and the final `FinalReport` for format, valid DES/DCS ranges, and existing supporting `evidence_id`s.

### Convergence

A Predict session converges when no new delegations remain or when the Reason step judges that another round would not change the conclusion. `SafetyNetHook` forces convergence if `max_rounds` or a timeout is reached.

## v1.0.0 scope

- Ingest: explore → convert → input with confirmation.
- Predict: dispatch → assess → delegate → synthesize → validate.
- Both workflows run through the hook system and return typed result dicts.

## Out of Scope

- **Benchmark workflow.** Evaluates Predict against evidence already in D-Base: matching records are temporarily marked `holdout-test`, Predict reads only `active` records so holdout records cannot leak into inference, predictions are compared against the held-out ground truth, and all holdout records are restored to `active` afterward — no temporary D-Base is created. Output is standard classification metrics plus the number of test records and rounds consumed. Deferred because v1.0.0 makes no accuracy demand; the `holdout-test` / `holdout-valid` statuses already exist in D-Base to support it (see `2.2_D-Base_storage_and_lifecycle.md`).
- **Train workflow.** Fine-tuning local LLMs on Dargus-produced Expert assessments, calibrating Bayesian and GNN models on held-out records, and learning delegation policies from historical convergence traces. Deferred until local-model deployment and post-v1.0.0 model integration are stable.
- **Multi-model prediction ensemble.** v1.0.0 Predict is Expert-driven. A future release can add specialized prediction primitives — `iris_search` (literature and database search), `iris_analog` (analog-based reasoning), `iris_bayes` (hierarchical Bayesian modeling), `iris_gnn` (graph-neural-network prediction), `iris_llm` (direct LLM reasoning) — aggregated by an `Iris.ensemble()` layer with weights derived from calibration and evidence coverage.
- **Advanced Routing Skills.** Expert knowledge graphs that traverse drug-target-disease relationships to find indirect evidence, contrastive retrieval that surfaces evidence challenging the current hypothesis, and temporal retrieval that prefers recent evidence or tracks its evolution.
