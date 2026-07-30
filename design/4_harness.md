# Agent Harness

> Dargus intelligence is organized as a small hierarchy of transparent Agents. Every Agent inherits the same harness, declares what it can use, and produces a typed report. This makes reasoning observable and predictions reviewable.

## Perceive → Reason → Act loop

`BaseAgent` is the abstract harness shared by all Agents. It implements a **Perceive → Reason → Act** (PRA) loop:

1. **Perceive** observes the current context: input, available evidence, tool outputs, and prior round results.
2. **Reason** decides what to do next: plan steps, choose tools, delegate, or conclude.
3. **Act** executes the chosen step and produces an observable output or side effect.

This is the only execution model for Agents. Every Agent — from Iris to the most specialized Domain Expert — follows this loop.

## Agent capability declaration

Each Agent subclass declares its capabilities through three explicit declarations:

- `PERMITTED_TOOLS` — which Tools it may call (a whitelist),
- `SUPPORTED_SKILLS` — which Skills it may execute,
- `SUPPORTED_LEVELS` — which biological levels it handles,
- `MAX_ROUNDS` — hard upper bound on loops.

At startup, each Agent validates that every Skill in `SUPPORTED_SKILLS` only requires Tools that the Agent is permitted to use. The Act step will only invoke Tools on the whitelist.

## Agent hierarchy

### Iris — top-level orchestrator

Iris is a BaseAgent whose job is to interpret user intent and dispatch to the appropriate workflow:

- Parse intent from user input (REPL or one-shot commands),
- For `predict` requests, invoke the Predict workflow,
- For `ingest` requests, invoke the Ingest workflow,
- Collect confirmations when human-in-the-loop is required.

Iris dispatches tasks to other Agents through agent communication protocols; it does not own or manage their lifecycle.

### Domain Experts

An Expert is a BaseAgent specialized for a subset of biological levels. Each Expert handles only the evidence within its domain:

| Expert | Handles |
|---|---|
| `MoleculeExpert` | `molecular`, `molecular-sim` |
| `BiomedExpert` | `cellular`–`animal` and `-sim` variants |
| `BioinfoExpert` | All levels, but only high-throughput / omics data |
| `ClinicExpert` | `rct`, `epi`, `rct-sim` |

### D4Expert — cross-domain synthesis

D4Expert handles all biological levels. Its role is cross-domain synthesis: combining Domain Expert reports into a single FinalReport, resolving contradictions, and producing the final DES ± DCS score pair.

## Delegation rules

When an Expert receives evidence outside its scope, it must create a `TaskDelegation` request rather than guessing. Core rules:

- A non-Bioinfo Expert seeing high-throughput data must delegate to `BioinfoExpert`.
- `BioinfoExpert` seeing non-omics data must delegate to the Expert matching the biological level.
- Level mismatch must follow the receiving Expert's delegation rules.

Delegation is the only mechanism for crossing domain boundaries — no Expert is permitted to assess evidence outside its `SUPPORTED_LEVELS`.

## Tool system

**Tools** are typed, executable capabilities registered in a `ToolRegistry`. Each Tool declares:

- name,
- parameter schema (`ToolParam` with name, type, required, default, description, enum),
- an `execute(...)` method.

Agents declare `PERMITTED_TOOLS` as a whitelist. The Act step only invokes Tools on that list. All D-Base reads and writes go through the single-writer D-Base Tools — no Agent writes evidence directly.

## Skill system

**Skills** are reusable methodologies defined as documents (markdown + YAML frontmatter) that describe multi-step processes. Each Skill declares:

- name and goal,
- `required_tools` (the Tools needed to execute the Skill),
- `supported_levels` (which biological levels the Skill operates on),
- input and output schemas.

A Skill is not executable by itself — it is a methodology that an Agent follows using its permitted Tools. Validation at startup ensures no Skill references a Tool the Agent cannot use.

## Hook system

**Hooks** are observer/callback functions registered at named points in the agent lifecycle. They allow cross-cutting concerns — session setup, safety limits, tool auditing, report validation — to live outside the Agent classes and outside any hardcoded workflow.

### Hook semantics

- Hooks receive a mutable **hook context** (a dictionary) and return it, possibly modified.
- A hook may raise an exception to stop the current round or session. By default a raised exception aborts the remaining hooks at that point and propagates to the runtime (**fail-closed**).
- Hooks may optionally declare themselves **observer-only**; observer-only hooks are logged and skipped if they raise (**fail-open**).
- The runtime owns and executes hooks. Agents do not invoke hooks directly.

### Hook context

The context passed to every hook is a dictionary with reserved keys including `session`, `round`, `agent`, `tools`, `task_spec`, `result`, `error`, and `report_valid`. Hooks may add their own namespaced keys prefixed with the hook name.

## Hook points

The PRA harness and the report flow expose the following hook points:

| Hook point | When it fires |
|---|---|
| `SESSION_START` | A workflow session begins |
| `PERCEIVE_START` | The Agent begins perceiving context for the round |
| `PERCEIVE_END` | The Agent has finished perception |
| `REASON_START` | The Agent begins reasoning/planning |
| `REASON_END` | The Agent has produced a plan or decision |
| `ACT_START` | The Agent begins tool/skill calls |
| `ACT_END` | The Agent has finished tool/skill calls |
| `ROUND_END` | The round is complete |
| `DOMAIN_REPORT_PRODUCED` | A Domain Expert has produced a report |
| `D4_REPORT_PRODUCED` | D4Expert has produced a synthesized report |
| `SESSION_END` | The workflow session ends |

### Core hook responsibilities

| Hook | Responsibility | Trigger point |
|---|---|---|
| `SessionInitHook` | Validate `task_spec.workflow` and initialize the session dictionary | `SESSION_START` |
| `SkeletonContextHook` | Inject round number, elapsed time, coverage, and pending delegations | `PERCEIVE_START` |
| `ToolAuditHook` | Record tool calls and enforce the workflow's tool allowlist | `ACT_END` |
| `SafetyNetHook` | Enforce max rounds, per-round timeout, and session timeout | `ROUND_END` |
| `ReportValidationHook` | Validate report format, DES/DCS range, and evidence_id existence | `DOMAIN_REPORT_PRODUCED`, `D4_REPORT_PRODUCED`, `SESSION_END` |
| `ResultReportHook` | Assemble the standardized result dict | `SESSION_END` |

### Hook registration

The runtime registers the core hooks in a fixed order at startup. Skills may register additional hooks at load time; by default they are appended after the core hooks for each point. Hooks at a given point run in registration order.
