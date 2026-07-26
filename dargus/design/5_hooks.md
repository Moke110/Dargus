# Hooks Design

> Hooks are observer/callback functions registered at named points in the agent lifecycle. They allow cross-cutting concerns—session setup, safety limits, tool auditing, report validation—to live outside the Agent classes and outside any hardcoded workflow. The runtime that owns and executes hooks is documented in `3_runtime.md`.

## Hook philosophy

Hooks receive a mutable **hook context** and return it, possibly modified. A hook may raise an exception to stop the current round or session. By default a raised exception aborts the remaining hooks at that point and propagates to the runtime (fail-closed). Hooks may optionally declare themselves observer-only; observer-only hooks are logged and skipped if they raise (fail-open).

Hooks influence the loop through the context. For example, `ReportValidationHook` can set a `report_valid` flag in the context; the deliver-report tool checks that flag and refuses to forward a report while it is false.

## Hook context

The context passed to every hook is a dictionary with the following reserved keys:

| Key | Content |
|---|---|
| `session` | Session dictionary created by `SessionInitHook` |
| `round` | Current round number and round state |
| `agent` | Identifier of the Agent whose loop is running |
| `tools` | Tool instances available to the Agent |
| `task_spec` | The workflow/task specification, including tool allowlists |
| `result` | Current output, if any |
| `error` | Current error, if any |
| `report_valid` | Set by `ReportValidationHook`; checked by deliver-report tools |

Hooks may add their own namespaced keys. Extra keys should be prefixed with the hook name to avoid collisions.

## Hook points

The Perceive → Reason → Act harness and the report flow expose the following hook points:

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

## Core hooks

The runtime registers the following core hooks in the order shown. They are defaults: a workflow can add hooks or override behavior, but removing a default hook requires explicit configuration.

| Hook | Responsibility | Trigger point |
|---|---|---|
| `SessionInitHook` | Validate `task_spec.workflow` and initialize the session dictionary | `SESSION_START` |
| `SkeletonContextHook` | Inject round number, elapsed time, coverage, and pending delegations | `PERCEIVE_START` |
| `ToolAuditHook` | Record tool calls and enforce the workflow's tool allowlist | `ACT_END` |
| `SafetyNetHook` | Enforce max rounds, per-round timeout, and session timeout | `ROUND_END` |
| `ReportValidationHook` | Validate report format, DES/DCS range, and evidence_id existence | `DOMAIN_REPORT_PRODUCED`, `D4_REPORT_PRODUCED`, `SESSION_END` |
| `ResultReportHook` | Assemble the standardized result dict | `SESSION_END` |

At `SESSION_END`, `ReportValidationHook` runs before `ResultReportHook`.

### ReportValidationHook

`ReportValidationHook` checks every intermediate and final report for:

1. correct report format/schema,
2. presence and valid range of `efficacy_score` (DES) and `confidence_score` (DCS) — waived when `confidence_level` is `insufficient_data`, in which case both scores must be unset,
3. existence in D-Base of every `evidence_id` cited in the report.

If validation fails, the hook sets `report_valid = false` in the context and raises `ReportValidationError` with a structured list of violations. The runtime routes the error back to the producing Agent. The deliver-report tool refuses to forward the report until `report_valid` is true again.

### SafetyNetHook

`SafetyNetHook` stops the loop when any of the following limits is reached:

- `max_rounds`: total PRA rounds for the session,
- `round_timeout`: wall-clock time allowed for one round,
- `session_timeout`: wall-clock time allowed for the whole session.

There is no minimum-evidence-coverage rule; some drug/endpoint pairs have very limited evidence.

## Why hooks instead of hardcoded workflows

The hook design inverts a traditional fixed-script approach: the same Agent harness executes any Skill, and hooks provide the workflow-specific policy. A new workflow is added by writing a Skill and registering the hooks it needs. This keeps Agent code free of workflow logic and makes new workflows composable rather than forked.

## Hook registration and ordering

The runtime registers the core hooks in a fixed order at startup. Skills register additional hooks at load time; by default they are appended after the core hooks for each point. Users can disable a core hook by name in `dargus_config.yaml` under a `hooks:` section. Hooks at a given point run in registration order.

## Hook execution semantics

- Hook execution is sequential.
- Each hook receives the current context and returns the context.
- A non-observer hook that raises aborts the remaining hooks at that point and propagates to the runtime, which stops the current round/session according to workflow policy.
- Observer-only hooks that raise are logged and skipped.
- Every hook invocation is recorded in a structured log (hook name, point, timestamp, elapsed time, success/failure).

## v1.0.0 scope

- Eleven hook points and the six core hooks above.
- Mutable hook context with reserved keys and exception-based veto.
- Structured hook observability logs.
