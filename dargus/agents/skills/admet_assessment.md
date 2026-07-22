---
name: admet_assessment
description: "ADMET property assessment from molecular and preclinical evidence"
version: "1.0.0"
supported_levels: [molecular, cellular, animal]
required_tools: [dbase_query, pubmed_search]
input_schema:
  drug_id: {type: string, required: true}
output_schema:
  absorption: {type: float}
  distribution: {type: float}
  metabolism: {type: float}
  excretion: {type: float}
  toxicity: {type: float}
  supporting_evidence: {type: array}
timeout_ms: 120000
fallback: skip
---
# ADMET Assessment

## When to Use
When evaluating drug-likeness and safety profile before clinical stages.

## Steps

1. **Query molecular ADME data**: Use `dbase_query` at `biological_level=molecular` for absorption/distribution/metabolism/excretion records.
2. **Query toxicity data**: Use `dbase_query` at `biological_level=cellular` and `animal` for toxicity records.
3. **Literature validation**: Use `pubmed_search` to find known ADMET concerns.
4. **Assemble output**: Score each ADMET dimension (0-1, higher = better).
