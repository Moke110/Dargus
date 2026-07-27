# Agent Design

> Dargus intelligence is organized as a small hierarchy of transparent Agents. Every Agent inherits the same harness, declares what it can use, and produces a typed report. This makes reasoning observable and predictions reviewable.

## BaseAgent harness

`BaseAgent` is the abstract harness shared by all Agents. It implements a **Perceive → Reason → Act** loop:

1. **Perceive** observes the current context: input, available evidence, tool outputs, and prior round results.
2. **Reason** decides what to do next: plan steps, choose tools, delegate, or conclude.
3. **Act** executes the chosen step and produces an observable output or side effect.

Each subclass declares:

- `PERMITTED_TOOLS` — which Tools it may call,
- `SUPPORTED_SKILLS` — which Skills it may execute,
- `SUPPORTED_LEVELS` — which biological levels it handles,
- `MAX_ROUNDS` — hard upper bound on loops.

The harness emits an `AgentReport` with rounds, convergence flag, confidence, findings, call traces, data gaps, and bias notes.

## Expert specialization

An `Expert` is a BaseAgent specialized for a subset of biological levels. v1.0.0 includes four Domain Experts and one cross-domain synthesis Expert:

| Expert | Handles | Focus |
|---|---|---|
| `MoleculeExpert` | `molecular`, `molecular-sim` | Drug chemistry, binding, physicochemical properties |
| `BiomedExpert` | `cellular`–`animal` and `-sim` variants | Preclinical biology, pharmacology, target-pathway reasoning |
| `BioinfoExpert` | All levels, but only high-throughput / omics data | Statistical power, batch effects, multiple-testing correction |
| `ClinicExpert` | `rct`, `epi`, `rct-sim` | Trial design, epidemiology, real-world evidence quality |
| `D4Expert` | All levels | Cross-domain synthesis, contradiction resolution, final confidence |

## Delegation rules

When an Expert receives evidence outside its scope, it creates a `TaskDelegation` request rather than guessing. Common cases:

- A non-Bioinfo Expert sees high-throughput data → delegate to `BioinfoExpert`.
- `BioinfoExpert` sees non-omics data → delegate to the Expert matching the biological level.
- Level mismatch → follow the receiving Expert’s `DELEGATION_RULES`.

## Iris, the top-level orchestrator

`Iris` is also a BaseAgent. Its job is to interpret user intent and dispatch to the appropriate workflow Skill:

- CLI input (REPL or one-shot `iris` command) → parse intent,
- `predict` → invoke Predict workflow,
- `ingest` → invoke Ingest workflow,
- collect confirmations when human-in-the-loop is required.

Iris does not centralize evidence retrieval. Each Expert uses its Routing Skill to fetch the evidence it needs from D-Base.

## v1.0.0 scope

- Shared Perceive → Reason → Act harness.
- Four Domain Experts + D4Expert with delegation rules.
- Iris as the intent-router and workflow launcher.
- Typed reports and call traces for every run.
- Human-in-the-loop confirmation hooks.

## Out of Scope

- **Knowledge permission layer.** A `PERMITTED_KNOWLEDGE` declaration controlling which Knowledge sources an Agent may query arrives with the Knowledge system (see `6_skills_tools_knowledge.md`).
- **Benchmark dispatch.** Iris routes `benchmark` requests to the Benchmark workflow once it ships (see `7_workflows.md`).
