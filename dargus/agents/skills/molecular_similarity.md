---
name: molecular_similarity
description: "Molecular similarity analysis against known reference compounds"
version: "1.0.0"
supported_levels: [molecular]
required_tools: [dbase_query]
input_schema:
  drug_id: {type: string, required: true}
  reference_compounds: {type: array, required: false}
output_schema:
  similarity_scores: {type: object}
  nearest_neighbors: {type: array}
  supporting_evidence: {type: array}
timeout_ms: 30000
fallback: skip
---
# Molecular Similarity

## When to Use
When comparing a drug to known reference compounds for mechanism inference or repurposing.

## Steps

1. **Query molecular records**: Use `dbase_query` at `biological_level=molecular` for the drug's structural and physicochemical data.
2. **Compare to references**: Compute similarity against provided reference compounds or known drug classes.
3. **Assemble output**: Return similarity scores and nearest neighbors.
