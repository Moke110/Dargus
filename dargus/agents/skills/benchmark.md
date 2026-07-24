---
name: benchmark
description: "Benchmark predict workflow against holdout ground truth"
version: "1.0.0"
supported_levels: [molecular, cellular, exvivo, animal, rct, epi]
required_tools: [dbase_query, dbase_write, dbase_status]
input_schema:
  holdout_ids: {type: array, required: true}
  metric: {type: string, required: false}
output_schema:
  accuracy: {type: float}
  precision: {type: float}
  recall: {type: float}
  f1: {type: float}
  n_test: {type: integer}
timeout_ms: 600000
fallback: skip
---
# Benchmark Skill

## Goal
Benchmark predict workflow accuracy against holdout ground truth.

## Workflow
1. Receive task_spec with holdout record IDs
2. Mark holdout records (exclude from query scope)
3. Run predict workflow using remaining active records
4. Compare predictions to holdout ground truth
5. Compute metrics: accuracy, precision, recall, f1
6. Restore holdout records to active state
7. Report BenchmarkResult with metrics
