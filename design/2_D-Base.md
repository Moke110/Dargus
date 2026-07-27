# D-Base Design

> D-Base is the **single, cumulative evidence store** at the heart of Dargus. It unifies evidence from every stage of drug development so that every prediction draws from the same authoritative pool.

## D-Base principles

1. **One variable for one evidence.** Each evidence record represents one experimental variable or manipulation. Compound results are decomposed into separate records so retrieval, assessment, and synthesis can reason about each claim precisely.

2. **One intact evidence for one input.** Every ingest input yields one complete, intact evidence record that preserves the source claim in full. Decomposition creates child records, but the original intact record remains the canonical source.

3. **LLM summary for fallback.** When structured extraction cannot fully populate the 50-field schema, an LLM-generated summary is stored as a fallback representation. The summary lives in a sidecar table keyed by `evidence_id` and can be regenerated or replaced without changing the evidence identity.

4. **Expandable vocabularies under human approval.** The controlled vocabularies for biological levels, evidence designs, endpoints, and other enumerations can be extended, but every new term requires explicit human approval before it is accepted into D-Base.

## Why one store matters

Drug-development evidence is usually scattered across spreadsheets, PDFs, databases, and notebooks. Dargus centralizes it into one keyed-object store with three consequences:

- **Reproducibility:** Every prediction can be traced to exact evidence records.
- **Accumulation:** Each `ingest` call grows institutional knowledge instead of creating yet another silo.
- **Consistency:** All Agents and models read the same schema and vocabularies.

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

All writes to D-Base go through a single component. No Agent, Expert, or workflow writes evidence directly. This guarantees:

- validation before persistence,
- content-addressed `evidence_id`,
- exact and semantic deduplication,
- embedding generation as a separate, re-runnable step (embeddings live outside records),
- append-only audit trail.

## Dual deduplication

D-Base uses two complementary deduplication strategies:

1. **Exact deduplication (hard gate).** A content hash of the identity fields produces `evidence_id`. If the same ID already exists, the incoming record is skipped. This is the default and cannot be bypassed.
2. **Semantic deduplication (soft flag).** The embedding tool converts the record to a dense vector and stores it in a separate embeddings sidecar keyed by `evidence_id` and the active embedding-model fingerprint. If cosine similarity to an existing record exceeds a threshold within the same drug/disease/endpoint scope, Dargus raises a `DuplicateReviewRequest`. The final decision is made by the human user, not by the store, any Agent, or an automated workflow.

## Sidecar fields live outside records

Three fields live outside the 50-field evidence record, each in its own append-only sidecar file keyed by `evidence_id`: the lifecycle `status` (with `superseded_by`), the LLM summary, and the embedding. Records stay stable and immutable; sidecar entries can be appended, regenerated, or superseded without altering evidence identity. See `2.2_D-Base_storage_and_lifecycle.md`.

## v1.0.0 scope

- 50-field record schema with controlled vocabularies, plus three sidecar fields (status, LLM summary, embedding) stored outside records.
- Exact + semantic deduplication.
- Semantic search over evidence embeddings.
- Separate sidecar tables with re-embedding support.
- Single-writer API with per-field updates and summary management.
- Append-only JSONL shards plus a derived Parquet view.

## Out of Scope

- **Schema extension (`D-Base.add_field()`).** A researcher-facing function to extend the 50-field schema with new fields. Schema changes affect validation, embedding, and vocabulary registries, so this requires a formal migration path.
- **Knowledge graph over D-Base.** D-Base v1.0.0 is a flat keyed-object store. A future release could layer a heterogeneous knowledge graph on top, linking drugs, targets, pathways, diseases, and trials while keeping D-Base as the authoritative evidence source.
