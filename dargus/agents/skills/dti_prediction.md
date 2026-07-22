---
name: dti_prediction
description: "Drug-target interaction prediction from molecular evidence"
version: "1.0.0"
supported_levels: [molecular]
required_tools: [dbase_query, pubmed_search]
input_schema:
  drug_id: {type: string, required: true}
  target_gene: {type: string, required: false}
output_schema:
  binding_score: {type: float}
  confidence: {type: float}
  supporting_evidence: {type: array}
timeout_ms: 60000
fallback: skip
---
# DTI Prediction

## When to Use
When molecular-level evidence is available and drug-target binding affinity needs to be estimated.

## Steps

1. **Query existing evidence**: Use `dbase_query` to search for known drug-target interactions for `drug_id` at `biological_level=molecular`.
2. **Validate with literature**: For any found interaction, use `pubmed_search` to find supporting publications.
3. **Assemble output**: Return `binding_score`, `confidence`, and `supporting_evidence` list with record IDs.

## Quality Notes
- Cross-reference with known off-target profiles when available
