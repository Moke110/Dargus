# D-Base Design

> D-Base is the **single, cumulative evidence store** at the heart of Dargus. It unifies evidence from every stage of drug development so that every prediction draws from the same authoritative pool.

## D-Base principles

1. **One variable for one evidence.** Each evidence record represents one experimental variable or manipulation. Compound results are decomposed into separate records so retrieval, assessment, and synthesis can reason about each claim precisely.

2. **One intact evidence for one input.** Every input yields one complete, intact evidence record that preserves the source claim in full. Decomposition creates child records, but the original intact record remains the canonical source.

3. **LLM summary for fallback.** When a record cannot be fully populated, an LLM-generated summary is stored as a fallback representation. The summary lives in a sidecar table keyed by `evidence_id` and can be regenerated or replaced without changing the evidence identity.

4. **Expandable vocabularies under human approval.** Controlled vocabularies for biological levels, evidence designs, endpoints, and other enumerations can be extended, but every new term requires explicit human approval before it is accepted into D-Base.

## The three-axis evidence model

Every evidence record is structured around three axes:

| Axis | Meaning | Example fields |
|---|---|---|
| `x` | The experimental variable | drug, gene, concentration, time, combination |
| `y` | The measured outcome | endpoint type, value, dispersion, statistics |
| `bg` | Background context shared by all measurements | disease, drugs, genes, model |

`xy.count` determines the shape of the record: `1` for descriptive, `2` for pairwise comparisons, `≥3` for sequential or dose-response data.

## Biological levels

D-Base records are tagged with one biological level. The canonical levels are:

- `molecular`, `molecular-sim`
- `cellular`, `cellular-sim`
- `exvivo`, `exvivo-sim`
- `animal`, `animal-sim`
- `rct`, `epi`, `rct-sim`

The `-sim` suffix marks simulation-derived evidence. `rct` and `epi` are clinical; all others are non-clinical.

## Single-writer invariant

All writes to D-Base go through a single component. No Agent or Expert writes evidence directly. This guarantees:

- validation before persistence,
- content-addressed `evidence_id`,
- exact and semantic deduplication,
- embedding generation as a separate, re-runnable step (embeddings live outside records),
- append-only audit trail.

## Dual deduplication

D-Base uses two complementary deduplication strategies:

1. **Exact deduplication (hard gate).** A content hash of the identity fields produces `evidence_id`. If the same ID already exists, the incoming record is skipped. This is the default and cannot be bypassed.
2. **Semantic deduplication (soft flag).** The store's embedding model converts the record to a dense vector. If cosine similarity to an existing record exceeds a threshold within the same experimental scope (`x` entity, `disease_id`, `y.type`), Dargus raises a `DuplicateReviewRequest`. The final decision is made by the human user, not by the store or any Agent.

## Identity and immutability

`evidence_id = ev_ + sha256(identity_fields)[:16]`. Once written, a record is immutable. Corrections are new records; the old record's sidecar status becomes `superseded` with `superseded_by` pointing at the new record.

## Sidecar fields live outside records

Three fields live outside the evidence record, each in its own append-only sidecar keyed by `evidence_id`: the lifecycle `status` (with `superseded_by`), the LLM summary, and the embedding. Sidecar entries can be appended, regenerated, or superseded without altering evidence identity — records stay stable and immutable. None of these fields participates in the `evidence_id` identity hash.

## Storage principles

- **Append-only authoritative storage.** Once written, evidence records are never modified or removed. All writes are append-only.
- **Rebuildable derived views.** Analytical views (e.g. Parquet) are derived from the authoritative store and can always be rebuilt.
- **Sidecar separation.** Mutable state (status, summary, embedding) lives in separate sidecar files keyed by `evidence_id`. Embedding sidecars are also keyed by model fingerprint to support model changes.
- **Re-embedding support.** When the project's embedding model changes, D-Base regenerates vectors with the new model and appends them to a new sidecar file. Old vectors are kept so switching back does not require recomputation.
- **Recovery.** The authoritative append-only store is the source of truth. Derived views and sidecars can be rebuilt or regenerated from it. If embedding generation is unavailable, evidence records are still written; semantic search simply skips records that have no vector for the active model.
