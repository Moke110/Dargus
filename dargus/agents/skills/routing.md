---
name: routing
description: "Evidence retrieval: field match + semantic embedding search over the active fingerprint sidecar"
version: "1.0.0"
supported_levels: [molecular, molecular-sim, cellular, cellular-sim, exvivo, exvivo-sim, animal, animal-sim, rct, epi, rct-sim]
required_tools: [dbase_query, embedding]
input_schema:
  query_text: {type: string, required: true}
  biological_level: {type: string, required: false}
  bg_drugs: {type: array, required: false}
  disease_id: {type: string, required: false}
  y_type: {type: string, required: false}
  top_k: {type: integer, required: false, default: 10}
output_schema:
  records: {type: array}
  scores: {type: array}
timeout_ms: 60000
fallback: empty_list
---
# Routing Skill

## Goal
Retrieve the evidence an Expert needs from D-Base during Predict — each Expert
fetches for itself, through its own domain lens, instead of Iris dispatching data.

## Method
1. **Field match.** Filter active D-Base records by `biological_level`,
   `bg.drugs` entity IDs, `bg.disease_id`, and `y.type`.
2. **Embed the query.** Ask the `embedding` tool (ToolCache-resident model)
   for the query vector.
3. **Rank.** Cosine-similarity against the vectors in the active
   embedding-model fingerprint sidecar (`sidecars/embeddings-{model_fp}.jsonl`).
   Records without a sidecar vector sort last.

## Constraints
- Only `active` records are eligible (status sidecar).
- Never re-embed records at query time; vectors come from the sidecar.
