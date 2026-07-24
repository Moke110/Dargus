---
name: predict
description: "Clinical efficacy prediction across drug-disease endpoints"
version: "1.0.0"
supported_levels: [molecular, cellular, exvivo, animal, rct, epi]
required_tools: [dbase_query, delegate_to_expert]
input_schema:
  drug_ids: {type: array, required: true}
  disease_id: {type: string, required: true}
  endpoints: {type: array, required: false}
output_schema:
  efficacy_low: {type: float}
  efficacy_up: {type: float}
  supporting_records: {type: array}
  overall_conclusion: {type: string}
timeout_ms: 300000
fallback: skip
---
# Predict Skill

## Goal
Assess the efficacy of a candidate drug for a target disease across clinical endpoints.

## Workflow
1. Receive task_spec with drug, disease, and endpoints
2. Query D-Base for relevant evidence using dbase_query
3. Delegate domain-specific assessment to DomainExperts:
   - molecular: molecular properties, drug-likeness
   - biomed: biological mechanism, pathway evidence
   - bioinfo: genomic/proteomic evidence
   - clinic: clinical trial data, epidemiological evidence
4. Collect ExpertReports from each delegation
5. Synthesize into unified D4Report with:
   - efficacy_low / efficacy_up (95% CI, both in [0,1])
   - supporting_records (list of D-Base record IDs)
   - overall_conclusion (text summary)
6. Submit FinalReport for acceptance gate validation

## Constraints
- Every supporting_record must exist in D-Base
- Efficacy intervals must be in [0,1]
- At least one DomainExpert must contribute evidence
- Report reasoning_mode for transparency
