# Workflow Design

> Dargus has three user-facing workflows: **Ingest**, **Predict**, and **Benchmark**. Each is implemented as a Skill-driven, hook-orchestrated process rather than a hardcoded script.

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

## Benchmark — evaluate Predict without data leakage

Benchmark measures Predict against evidence already in D-Base.

### Rules

- Matching records are temporarily marked `holdout-test`.
- Predict reads only `active` records, so holdout records cannot leak into inference.
- Predictions are compared against the held-out ground truth.
- After Benchmark finishes, all holdout records are restored to `active`.
- No temporary D-Base is created.

### Output

Benchmark reports standard classification metrics plus the number of test records and the number of rounds consumed.

## v1.0.0 scope

- Ingest: explore → convert → input with confirmation.
- Predict: dispatch → assess → delegate → synthesize → validate.
- Benchmark: holdout marking, active-only inference, status restoration.
- All three workflows run through the hook system and return typed result dicts.
