# Agent Harness

> Dargus intelligence is organized as a small hierarchy of transparent Agents. Every Agent inherits the same harness, declares what it can use, and produces a typed report. This makes reasoning observable.

## Perceive → Reason → Act loop

`BaseAgent` is the abstract harness shared by all Agents. It implements a **Perceive → Reason → Act** (PRA) loop:

1. **Perceive** assembles context without an LLM call: the agent's `system_prompt`, the definitions of its `PERMITTED_TOOLS` (as JSON Schema), skill content, and the Conversation projected to LLM messages.
2. **Reason** forwards the perceive cache to the LLM. The LLM returns a JSON response — `{"action": "text", "text": "..."}` for dialogue, or `{"action": "tool_call", "tool": "...", "params": {...}}` for a tool call. This is the only LLM call per round.
3. **Act** authorizes the requested Tool against `PERMITTED_TOOLS`, executes it, and settles the result into the Conversation. A text response ends the loop; a tool_call continues it.

This is the only execution model for Agents. Every Agent — from Iris to the most specialized Domain Expert — follows this loop. Convergence is LLM-decided: a text response ends the loop, a tool_call continues it, bounded by `MAX_ROUNDS`.

## Agent capability declaration

Each Agent subclass declares its capabilities:

- `name` — the agent's identity,
- `system_prompt` — the LLM system prompt (a class attribute, replacing the removed per-mode system prompts),
- `PERMITTED_TOOLS` — which Tools it may call (a whitelist),
- `SUPPORTED_SKILLS` — which Skills it may execute,
- `SUPPORTED_LEVELS` — which biological levels it handles (Experts),
- `MAX_ROUNDS` — hard upper bound on loops.

At startup, each Agent validates that every Skill in `SUPPORTED_SKILLS` only requires Tools that the Agent is permitted to use. The Act step only invokes Tools on the whitelist.

## Agent hierarchy

### Iris — top-level orchestrator

Iris is a BaseAgent and the primary interface for dialogue. It converses with the user and can call Tools (currently `read_file`). Task-specific orchestration (predict/ingest dispatch) was removed with the task-specific code; Iris keeps the PRA dialogue loop and D-Base status reporting.

### Domain Experts

An Expert is a BaseAgent specialized for a subset of biological levels. Each Expert handles only the evidence within its domain:

| Expert | Handles |
|---|---|
| `MoleculeExpert` | `molecular`, `molecular-sim` |
| `BiomedExpert` | `cellular`–`animal` and `-sim` variants |
| `BioinfoExpert` | All levels, but only high-throughput / omics data |
| `ClinicExpert` | `rct`, `epi`, `rct-sim` |

Expert assessment logic (the `assess()` method and per-domain scoring) was removed with the task-specific code. Experts currently exist as skeletons declaring their domain scope (`SUPPORTED_LEVELS`, `system_prompt`, empty `PERMITTED_TOOLS`), awaiting the redo.

### D4Expert — cross-domain synthesis

D4Expert handles all biological levels. Its role is cross-domain synthesis: combining Domain Expert reports into a single final report and resolving contradictions. Its synthesis methods were removed with the task-specific code; it currently exists as a skeleton.

## Tool system

**Tools** are typed, executable capabilities registered in a `ToolRegistry`. Each Tool declares:

- name,
- parameter schema (`ToolParam` with name, type, required, default, description, enum),
- an `execute(...)` method.

Agents declare `PERMITTED_TOOLS` as a whitelist. The Act step only invokes Tools on that list. Currently the runtime registers two general-purpose file Tools (`read_file`, `write_file`) wired to the WorkspaceGuard; task-specific Tools (D-Base query, PubMed search) were removed with the task-specific code.

## Skill system

**Skills** are reusable methodologies defined as documents (markdown + YAML frontmatter) that describe multi-step processes. Each Skill declares:

- name and goal,
- `required_tools` (the Tools needed to execute the Skill),
- `supported_levels` (which biological levels the Skill operates on),
- input and output schemas.

A Skill is not executable by itself — it is a methodology that an Agent follows using its permitted Tools. Validation at startup ensures no Skill references a Tool the Agent cannot use. The task-specific Skills were removed with the task-specific code; the SkillRegistry remains as the loading mechanism.
